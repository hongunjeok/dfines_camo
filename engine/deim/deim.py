"""
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import torch.nn as nn
from ..core import register


__all__ = ['DEIM', ]


@register()
class DEIM(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        

    def forward(self, x, targets=None):
        feats = self.backbone(x)                # list/tuple of feats
        feats = self.encoder(feats)             # HybridEncoder.forward -> outs (list)
        outputs = self.decoder(feats, targets)  # dict (pred_logits, pred_boxes, ...)

        # ★ 여기서 sem_feats를 outputs에 실어 보냄 (학습 시에만)
        if self.training and getattr(self.encoder, "last_sem_feats", None) is not None:
            outputs["sem_feats"] = self.encoder.last_sem_feats
            

        return outputs

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
