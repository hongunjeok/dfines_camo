"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""

import wandb
import sys
import math
from typing import Iterable

import torch
import torch.amp
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils
import numpy as np  # NEW

wandb.init(project='Camo_small', config={'dataset' : 'camo'})
config = wandb.config

def train_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler :Warmup = kwargs.get('lr_warmup_scheduler', None)

    cur_iters = epoch * len(data_loader)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)

            if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
                print(outputs['pred_boxes'])
                state = model.state_dict()
                new_state = {}
                for key, value in model.state_dict().items():
                    # Replace 'module' with 'model' in each key
                    new_key = key.replace('module.', '')
                    # Add the updated key-value pair to the state dictionary
                    state[new_key] = value
                new_state['model'] = state
                dist_utils.save_on_master(new_state, "./NaN.pth")

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)

            loss : torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        # ema
        if ema is not None:
            ema.update(model)

        if self_lr_scheduler:
            optimizer = lr_scheduler.step(cur_iters + i, optimizer)
        else:
            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # WandB 로깅 추가
        if dist_utils.is_main_process() and global_step % 10 == 0:
            wandb.log({"Loss/total": loss_value.item(), "epoch": epoch, "step": global_step})
            for j, pg in enumerate(optimizer.param_groups):
                wandb.log({f'Lr/pg_{j}': pg['lr']})
            for k, v in loss_dict_reduced.items():
                wandb.log({f'Loss/{k}': v.item(), "epoch": epoch, "step": global_step})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device, epoch: int = None):  # CHANGED
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    iou_types = coco_evaluator.iou_types

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessor(outputs, orig_target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    # ✅ WandB 로깅 추가 (AP, AR 기록)
    if dist_utils.is_main_process():
        bbox_stats = stats.get('coco_eval_bbox', [])  # stats에서 bbox key 가져오고, 없으면 빈 리스트 반환
        
        wandb.log({
            "Evaluation/AP (IoU=0.50:0.95, all)": bbox_stats[0] if len(bbox_stats) > 0 else 0,  # AP 0.50:0.95
            "Evaluation/AP (IoU=0.50, all)": bbox_stats[1] if len(bbox_stats) > 1 else 0,  # AP 0.50
            "Evaluation/AP (IoU=0.75, all)": bbox_stats[2] if len(bbox_stats) > 2 else 0,  # AP 0.75
            "Evaluation/AP (IoU=0.50:0.95, small)": bbox_stats[3] if len(bbox_stats) > 3 else 0,  # AP small
            "Evaluation/AP (IoU=0.50:0.95, medium)": bbox_stats[4] if len(bbox_stats) > 4 else 0,  # AP medium
            "Evaluation/AP (IoU=0.50:0.95, large)": bbox_stats[5] if len(bbox_stats) > 5 else 0,  # AP large
            "Evaluation/AR (IoU=0.50:0.95, maxDets=1)": bbox_stats[6] if len(bbox_stats) > 6 else 0,  # AR maxDets=1
            "Evaluation/AR (IoU=0.50:0.95, maxDets=10)": bbox_stats[7] if len(bbox_stats) > 7 else 0,  # AR maxDets=10
            "Evaluation/AR (IoU=0.50:0.95, maxDets=100)": bbox_stats[8] if len(bbox_stats) > 8 else 0,  # AR maxDets=100
            "Evaluation/AR (IoU=0.50:0.95, small)": bbox_stats[9] if len(bbox_stats) > 9 else 0,  # AR small
            "Evaluation/AR (IoU=0.50:0.95, medium)": bbox_stats[10] if len(bbox_stats) > 10 else 0,  # AR medium
            "Evaluation/AR (IoU=0.50:0.95, large)": bbox_stats[11] if len(bbox_stats) > 11 else 0,  # AR large
        })

    # === [NEW] 클래스별 AP 로깅 ===
    if dist_utils.is_main_process() and coco_evaluator is not None and 'bbox' in iou_types:
        ce = coco_evaluator.coco_eval['bbox']           # pycocotools.cocoeval.COCOeval
        precisions = ce.eval['precision']               # shape: [T, R, K, A, M]
        if precisions is not None:
            iou_thrs = ce.params.iouThrs                # 길이 T
            # area=all, maxDets=100 인덱스 선택
            aind = 0
            if hasattr(ce.params, 'areaRngLbl') and 'all' in ce.params.areaRngLbl:
                aind = ce.params.areaRngLbl.index('all')
            mind = len(ce.params.maxDets) - 1
            if 100 in ce.params.maxDets:
                mind = ce.params.maxDets.index(100)
            # IoU=0.50 인덱스
            t50 = int(np.where(np.isclose(iou_thrs, 0.5))[0][0])

            # 카테고리 이름들
            cats = ce.cocoGt.loadCats(ce.params.catIds)
            names = [c['name'] for c in cats]

            class_logs = {}
            for k, name in enumerate(names):
                # AP@[.50:.95]: 모든 IoU/recall 평균
                p = precisions[:, :, k, aind, mind]
                p = p[p > -1]
                ap = float(np.mean(p)) if p.size else 0.0

                # AP@.50: IoU=0.5 슬라이스만 평균
                p50 = precisions[t50, :, k, aind, mind]
                p50 = p50[p50 > -1]
                ap50 = float(np.mean(p50)) if p50.size else 0.0

                class_logs[f"ClassAP/AP@[.50:.95]/{name}"] = ap
                class_logs[f"ClassAP/AP@.50/{name}"] = ap50

            if epoch is not None:
                class_logs["epoch"] = epoch
            wandb.log(class_logs)


    return stats, coco_evaluator
