"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import torchvision
import math

import copy

from .dfine_utils import bbox2distance
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from ..misc.dist_utils import get_world_size, is_dist_available_and_initialized
from ..core import register

def second_order_proto(feat, mask, iters=3, eps=1e-6, groups=1):
    """
    feat: [B,C,H,W]  (may arrive in fp16 from autocast)
    mask: [B,1,H,W]
    returns: [B,C] (float32, L2-normalized)
    """
    # 🔧 dtype 정규화 (수치 안정 + bmm dtype mismatch 방지)
    feat = feat.float()
    mask = mask.float()

    B, C, H, W = feat.shape
    N = H * W

    x = feat.view(B, C, N)            # [B,C,N], float32
    w = mask.view(B, 1, N)            # [B,1,N], float32
    wsum = w.sum(dim=2, keepdim=True).clamp(min=1.0)
    w = w / wsum

    if groups <= 1:
        v = (x * w).sum(dim=2)                 # [B,C]
        v = F.normalize(v, dim=1, eps=eps)
        for _ in range(iters):
            xv = torch.bmm(x.transpose(1, 2), v.unsqueeze(-1)).squeeze(-1)     # [B,N]
            y  = torch.bmm(x, (w.squeeze(1) * xv).unsqueeze(-1)).squeeze(-1)   # [B,C]
            v  = F.normalize(y, dim=1, eps=eps)
        return v
    else:
        assert C % groups == 0
        gc = C // groups
        vs = []
        for g in range(groups):
            xs = x[:, g*gc:(g+1)*gc, :]                                         # [B,gc,N]
            vg = (xs * w).sum(dim=2)                                            # [B,gc]
            vg = F.normalize(vg, dim=1, eps=eps)
            for _ in range(iters):
                xsv = torch.bmm(xs.transpose(1, 2), vg.unsqueeze(-1)).squeeze(-1)         # [B,N]
                yg  = torch.bmm(xs, (w.squeeze(1) * xsv).unsqueeze(-1)).squeeze(-1)       # [B,gc]
                vg  = F.normalize(yg, dim=1, eps=eps)
            vs.append(vg)
        v = torch.cat(vs, dim=1)                                                # [B,C]
        return F.normalize(v, dim=1, eps=eps)

def masked_avg_pool(feat, mask):
    # feat: [B,C,H,W], mask: [B,1,H,W] (0/1)
    area = mask.sum(dim=(2,3), keepdim=True).clamp(min=1.0)
    pooled = (feat * mask).sum(dim=(2,3), keepdim=True) / area
    return pooled.squeeze(-1).squeeze(-1)  # [B,C]

def rasterize_box_mask_xyxy(boxes_xyxy, H, W, device):
    # 여러 박스 union 바이너리 마스크 (절대좌표 xyxy 기준)
    m = torch.zeros((1, H, W), dtype=torch.float32, device=device)
    if boxes_xyxy.numel() == 0:
        return m
    x1y1 = boxes_xyxy[:, :2].floor().clamp(min=0)
    x2y2 = boxes_xyxy[:, 2:].ceil()
    x2y2[:,0] = x2y2[:,0].clamp(max=W)
    x2y2[:,1] = x2y2[:,1].clamp(max=H)
    for (x1,y1),(x2,y2) in zip(x1y1.long(), x2y2.long()):
        if x2 > x1 and y2 > y1:
            m[:, y1:y2, x1:x2] = 1.0
    return m

@register()
class DEIMCriterion(nn.Module):
    """ This class computes the loss for DEIM.
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, \
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.5,
        num_classes=80,
        reg_max=32,
        boxes_weight_format=None,
        share_matched_indices=False,
        mal_alpha=None,
        use_uni_set=True,
        ):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            num_classes: number of object categories, omitting the special no-object category.
            reg_max (int): Max number of the discrete bins in D-FINE.
            boxes_weight_format: format for boxes weight (iou, ).
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.reg_max = reg_max
        self.num_pos, self.num_neg = None, None
        self.mal_alpha = mal_alpha
        self.use_uni_set = use_uni_set
        self.lambda_sem = weight_dict.get('lambda_sem', None)
        # 혹은 고정값으로 시작
        # self.lambda_sem = 0.1
        # 입력 박스 포맷 플래그(데이터셋에 맞게)
        self.box_format = 'xyxy'  # 'xyxy' 절대좌표가 기본. 필요시 yaml로 빼세요.
        
    def loss_sem_proto(self, outputs, targets, indices, num_boxes, **kwargs):
        """
        - PAN 출력(outs)에서 넘어온 outputs['sem_feats'](p_low, p_high)를 사용한다는 전제.
        - 스케일별 top-k 선택 마스크를 mask_sels에 저장하여 이후 최소 픽셀 가드에 활용.
        - 워밍업→코사인 디케이 스케줄은 kwargs로 전달되는 global_step / total_steps를 사용.
        """
        wd = self.weight_dict
        proto_detach = str(wd.get('proto_detach', 'always'))   # "early"|"always"|"never"
        warmup = int(wd.get('sem_warmup', 800))                # 기존 default=800 유지

        # 스케줄용 글로벌 스텝 (엔진에서 kwargs로 넘겨주고 있음)
        gs = int(kwargs.get('global_step', -1))        

        if ('sem_feats' not in outputs) or (outputs['sem_feats'] is None):
            return {}

        sem_feats = outputs['sem_feats']
        if not isinstance(sem_feats, (list, tuple)):
            sem_feats = [sem_feats]
        assert len(sem_feats) >= 1, "outputs['sem_feats'] must contain at least the anchor p_high."
        # 앵커(글로벌) 정의 및 나머지 스케일 분리
        p_high = sem_feats[0]                 # anchor (global)
        others = list(sem_feats[1:])          # align 대상들
        B, C = p_high.shape[:2]

        # 앵커 + 나머지 스케일들 순회
        feats_all = [p_high] + others
        dims = []  # [(Hs, Ws), ...]

        protos = []            # 스케일별 프로토타입
        valid_any = []         # 각 스케일 FG 존재 여부(배치별)
        mask_sels = []         # 각 스케일의 top-k 선택 마스크

        for feat in feats_all:
            Hs, Ws = feat.shape[-2:]
            dims.append((Hs, Ws))
            masks_b, has_fg_b = [], []

            # ----- 1) GT box → 스케일 해상도 마스크로 변환 -----
            for i in range(B):
                # (orig_size는 tensor이므로 .tolist()로 안전 변환)
                H_img, W_img = map(int, targets[i]['orig_size'].tolist())
                boxes = targets[i]['boxes']  # [Ni,4]; 형식이 xyxy 또는 cxcywh일 수 있음
                boxes_xyxy = boxes

                if boxes.numel() > 0:
                    if float(boxes.max()) <= 1.5:  # 정규화 cxcywh로 간주
                        b = boxes * boxes.new_tensor([W_img, H_img, W_img, H_img])
                        x_c, y_c, w, h = b.unbind(-1)
                        x1 = (x_c - w / 2).clamp(0, W_img)
                        y1 = (y_c - h / 2).clamp(0, H_img)
                        x2 = (x_c + w / 2).clamp(0, W_img)
                        y2 = (y_c + h / 2).clamp(0, H_img)
                        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
                    else:
                        # 좌표 유효성 대략 검사 → 잘못된 형식이면 cxcywh로 처리
                        xy_ok = (
                            ((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))
                            .float()
                            .mean()
                            .item()
                            if boxes.shape[0] > 0
                            else 1.0
                        )
                        if xy_ok < 0.5:
                            x_c, y_c, w, h = boxes.unbind(-1)
                            x1 = (x_c - w / 2).clamp(0, W_img)
                            y1 = (y_c - h / 2).clamp(0, H_img)
                            x2 = (x_c + w / 2).clamp(0, W_img)
                            y2 = (y_c + h / 2).clamp(0, H_img)
                            boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

                m_full = rasterize_box_mask_xyxy(boxes_xyxy, H_img, W_img, feat.device)   # [1,H_img,W_img]
                m_s = F.interpolate(m_full.unsqueeze(0), size=(Hs, Ws), mode='nearest').squeeze(0)  # [1,Hs,Ws]
                masks_b.append(m_s)
                has_fg_b.append(m_s.sum() > 0)

            mask = torch.stack(masks_b, dim=0)  # [B,1,Hs,Ws]
            has_fg_b = torch.tensor(has_fg_b, device=feat.device, dtype=torch.bool)

            # ----- 2) top-k 픽셀 선택 (연속형 비율 + 음수 무한대 마스킹) -----
            with torch.no_grad():
                feat32 = feat.float()
                mask32 = mask.float()

                # 점수: ||feat|| (L2 norm)
                score = feat32.pow(2).sum(1, keepdim=True).sqrt() * mask32  # [B,1,Hs,Ws]

                B_, _, Hs_, Ws_ = score.shape
                flat_score = score.flatten(2)   # [B,1,Hs*Ws]
                flat_mask = mask32.flatten(2)   # [B,1,Hs*Ws]

                # 배경은 매우 작은 값으로 마스킹
                neg_inf = torch.finfo(flat_score.dtype).min
                flat_score = flat_score.masked_fill(flat_mask < 0.5, neg_inf)

                sel = torch.zeros_like(flat_score)  # [B,1,Hs*Ws]

                # 샘플별 k 계산
                for i in range(B_):
                    fg_i = int((flat_mask[i] >= 0.5).sum().item())
                    if fg_i <= 0:
                        continue

                    # rho = rho_min + (rho_max - rho_min) * (1 - exp(-fg_i / T))
                    rho_min, rho_max, T = 0.10, 0.18, 96.0
                    # 텐서→스칼라 계산(정확도/속도 모두 충분)
                    rho = float(
                        rho_min + (rho_max - rho_min) * (1.0 - math.exp(-fg_i / T))
                    )
                    k_floor, k_cap = 12, 0.20  # 최소 픽셀/격자 상한(20%)
                    k_i = max(int(fg_i * rho), k_floor)
                    k_i = min(k_i, fg_i)
                    k_i = min(k_i, int(k_cap * Hs_ * Ws_))
                    if k_i <= 0:
                        continue

                    # top-k 인덱스 선택
                    _, idx_i = torch.topk(flat_score[i : i + 1], k=k_i, dim=2)
                    sel[i : i + 1].scatter_(2, idx_i, 1.0)

                # 선택 마스크 복원 + 한 번 dilate
                mask_sel = sel.view_as(score)                    # [B,1,Hs,Ws]
                mask_sel = F.max_pool2d(mask_sel, 3, 1, 1)      # 3x3 dilate 1회

            # ----- 3) 프로토타입 계산 (avg + 2nd-order), 정규화 -----
            g = 16 if (feat.shape[1] >= 128 and feat.shape[1] % 4 == 0) else 1
            proto1 = second_order_proto(feat, mask_sel, iters=2, eps=1e-6, groups=g)
            proto0 = F.normalize(masked_avg_pool(feat.float(), mask_sel), dim=1, eps=1e-6)
            alpha = 0.8
            proto = F.normalize(alpha * proto1 + (1 - alpha) * proto0, dim=1, eps=1e-6)

            protos.append(proto)            # [B,C]
            valid_any.append(has_fg_b)
            mask_sels.append(mask_sel)      # [B,1,Hs,Ws]

        # ----- 최소 픽셀 가드 & 단계적 정렬 (feat2→feat1, feat1→feat0) -----
        cnts = [m.sum(dim=(2, 3)).squeeze(1) for m in mask_sels]
        mins = [max(4, int(0.005 * Hs * Ws)) for (Hs, Ws) in dims]

        pair_losses = []
        # feats_all/protos 순서가 [feat2, feat1, feat0]일 때 → (1,0), (2,1)
        pair_indices = [(i_low, i_high) for i_low, i_high in zip(range(1, len(protos)), range(0, len(protos)-1))]
        for i_low, i_high in pair_indices:
            valid = (valid_any[i_low] & valid_any[i_high] &
                     (cnts[i_low] >= mins[i_low]) & (cnts[i_high] >= mins[i_high]))
            if valid.any():
                # 상위 스케일을 앵커로 고정(detach) → 한 단계 아래 스케일이 앵커를 따라가도록
                p_anchor_raw = F.normalize(protos[i_high][valid].clamp(-1, 1), dim=1, eps=1e-6)
                if proto_detach == 'never':
                    p_anchor = p_anchor_raw
                elif proto_detach == 'early':
                    # warmup 동안만 detach, 그 이후엔 그래디언트 전파 허용
                    if gs >= 0 and gs >= warmup:
                        p_anchor = p_anchor_raw
                    else:
                        p_anchor = p_anchor_raw.detach()
                else:  # 'always'
                    p_anchor = p_anchor_raw.detach()
                p_curr   = F.normalize(protos[i_low][valid].clamp(-1, 1), dim=1, eps=1e-6)
                cos_sim  = (p_anchor * p_curr).sum(dim=1)
                pair_losses.append((1.0 - cos_sim).mean())

        loss = torch.stack(pair_losses).mean() if pair_losses else feats_all[0].new_tensor(0.0)

        gs = int(kwargs.get('global_step', -1))
        warmup = 800
        scale = 1.0 if gs < 0 else float(min(1.0, gs / max(1, warmup)))
        return {'loss_sem_proto': loss * scale}

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, **kwargs):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None, **kwargs):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None, **kwargs):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        target_score = target_score.pow(self.gamma)
        if self.mal_alpha != None:
            weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target
        else:
            weight = pred_score.pow(self.gamma) * (1 - target) + target

        # print(" ### DEIM-gamma{}-alpha{} ### ".format(self.gamma, self.mal_alpha))
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_mal': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None, **kwargs):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(\
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_local(self, outputs, targets, indices, num_boxes, T=5, **kwargs):
        """Compute Fine-Grained Localization (FGL) Loss
            and Decoupled Distillation Focal (DDF) Loss. """

        losses = {}
        if 'pred_corners' in outputs:
            idx = self._get_src_permutation_idx(indices)
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            pred_corners = outputs['pred_corners'][idx].reshape(-1, (self.reg_max+1))
            ref_points = outputs['ref_points'][idx].detach()
            with torch.no_grad():
                if self.fgl_targets_dn is None and 'is_dn' in outputs:
                        self.fgl_targets_dn= bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),
                                                        self.reg_max, outputs['reg_scale'], outputs['up'])
                if self.fgl_targets is None and 'is_dn' not in outputs:
                        self.fgl_targets = bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),
                                                        self.reg_max, outputs['reg_scale'], outputs['up'])

            target_corners, weight_right, weight_left = self.fgl_targets_dn if 'is_dn' in outputs else self.fgl_targets

            ious = torch.diag(box_iou(\
                        box_cxcywh_to_xyxy(outputs['pred_boxes'][idx]), box_cxcywh_to_xyxy(target_boxes))[0])
            weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

            losses['loss_fgl'] = self.unimodal_distribution_focal_loss(
                pred_corners, target_corners, weight_right, weight_left, weight_targets, avg_factor=num_boxes)

            if 'teacher_corners' in outputs:
                pred_corners = outputs['pred_corners'].reshape(-1, (self.reg_max+1))
                target_corners = outputs['teacher_corners'].reshape(-1, (self.reg_max+1))
                if not torch.equal(pred_corners, target_corners):
                    weight_targets_local = outputs['teacher_logits'].sigmoid().max(dim=-1)[0]

                    mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                    mask[idx] = True
                    mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                    weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(weight_targets_local.dtype)
                    weight_targets_local = weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

                    loss_match_local = weight_targets_local * (T ** 2) * (nn.KLDivLoss(reduction='none')
                    (F.log_softmax(pred_corners / T, dim=1), F.softmax(target_corners.detach() / T, dim=1))).sum(-1)
                    if 'is_dn' not in outputs:
                        batch_scale = 8 / outputs['pred_boxes'].shape[0]  # Avoid the influence of batch size per GPU
                        self.num_pos, self.num_neg = (mask.sum() * batch_scale) ** 0.5, ((~mask).sum() * batch_scale) ** 0.5
                    loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
                    loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
                    losses['loss_ddf'] = (loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg) / (self.num_pos + self.num_neg)

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_go_indices(self, indices, indices_aux_list):
        """Get a matching union set across all decoder layers. """
        results = []
        for indices_aux in indices_aux_list:
            indices = [(torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                        for idx1, idx2 in zip(indices.copy(), indices_aux.copy())]

        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    def _clear_cache(self):
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.num_pos, self.num_neg = None, None

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            'mal': self.loss_labels_mal,
            'local': self.loss_local,
            'sem_proto': self.loss_sem_proto,   # ← 추가
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)['indices']
        self._clear_cache()

        # Get the matching union set across all decoder layers.
        if 'aux_outputs' in outputs:
            indices_aux_list, cached_indices, cached_indices_enc = [], [], []
            aux_outputs_list = outputs['aux_outputs']
            if 'pre_outputs' in outputs:
                aux_outputs_list = outputs['aux_outputs'] + [outputs['pre_outputs']]
            for i, aux_outputs in enumerate(aux_outputs_list):
                indices_aux = self.matcher(aux_outputs, targets)['indices']
                cached_indices.append(indices_aux)
                indices_aux_list.append(indices_aux)
            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                indices_enc = self.matcher(aux_outputs, targets)['indices']
                cached_indices_enc.append(indices_enc)
                indices_aux_list.append(indices_enc)
            indices_go = self._get_go_indices(indices, indices_aux_list)

            num_boxes_go = sum(len(x[0]) for x in indices_go)
            num_boxes_go = torch.as_tensor([num_boxes_go], dtype=torch.float, device=next(iter(outputs.values())).device)
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes_go)
            num_boxes_go = torch.clamp(num_boxes_go / get_world_size(), min=1).item()

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses, main loss
        losses = {}
        for loss in self.losses:
            # TODO, indices and num_box are different from RT-DETRv2
            use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
            indices_in = indices_go if use_uni_set else indices
            num_boxes_in = num_boxes_go if use_uni_set else num_boxes
            meta = self.get_loss_meta_info(loss, outputs, targets, indices_in)
            l_dict = self.get_loss(loss, outputs, targets, indices_in, num_boxes_in, **meta, **kwargs)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if 'local' in self.losses:      # only work for local loss
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    # TODO, indices and num_box are different from RT-DETRv2
                    use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
                    indices_in = indices_go if use_uni_set else cached_indices[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta, **kwargs)

                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of auxiliary traditional head output at first decoder layer. just for dfine
        if 'pre_outputs' in outputs:
            aux_outputs = outputs['pre_outputs']
            for loss in self.losses:
                # TODO, indices and num_box are different from RT-DETRv2
                use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
                indices_in = indices_go if use_uni_set else cached_indices[-1]
                num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta, **kwargs)

                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                l_dict = {k + '_pre': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # In case of encoder auxiliary losses.
        if 'enc_aux_outputs' in outputs:
            assert 'enc_meta' in outputs, ''
            class_agnostic = outputs['enc_meta']['class_agnostic']
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                for loss in self.losses:
                    # TODO, indices and num_box are different from RT-DETRv2
                    use_uni_set = self.use_uni_set and (loss == 'boxes')
                    indices_in = indices_go if use_uni_set else cached_indices_enc[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices_in, num_boxes_in, **meta, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if class_agnostic:
                self.num_classes = orig_num_classes

        # In case of cdn auxiliary losses.
        if 'dn_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices_dn = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']

            for i, aux_outputs in enumerate(outputs['dn_outputs']):
                if 'local' in self.losses:
                    aux_outputs['is_dn'] = True
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)  # ← 추가
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # In case of auxiliary traditional head output at first decoder layer, just for dfine
            if 'dn_pre_outputs' in outputs:
                aux_outputs = outputs['dn_pre_outputs']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + '_dn_pre': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # For debugging Objects365 pre-train.
        losses = {k:torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}
        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs['pred_boxes'][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == 'iou':
            iou, _ = box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes))
            iou = torch.diag(iou)
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou(\
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)))
        else:
            raise AttributeError()

        if loss in ('boxes', ):
            meta = {'boxes_weight': iou}
        elif loss in ('vfl', 'mal'):
            meta = {'values': iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices
        """
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))

        return dn_match_indices


    def feature_loss_function(self, fea, target_fea):
        loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
        return torch.abs(loss)


    def unimodal_distribution_focal_loss(self, pred, label, weight_right, weight_left, weight=None, reduction='sum', avg_factor=None):
        dis_left = label.long()
        dis_right = dis_left + 1

        loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left.reshape(-1) \
             + F.cross_entropy(pred, dis_right, reduction='none') * weight_right.reshape(-1)

        if weight is not None:
            weight = weight.float()
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        return loss

    def get_gradual_steps(self, outputs):
        num_layers = len(outputs['aux_outputs']) + 1 if 'aux_outputs' in outputs else 1
        step = .5 / (num_layers - 1)
        opt_list = [.5  + step * i for i in range(num_layers)] if num_layers > 1 else [1]
        return opt_list
