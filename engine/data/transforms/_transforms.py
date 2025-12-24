"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

import PIL
import PIL.Image

from typing import Any, Dict, List, Optional

from .._misc import convert_to_tv_tensor, _boxes_keys
from .._misc import Image, Video, Mask, BoundingBoxes
from .._misc import SanitizeBoundingBoxes

from ...core import register
torchvision.disable_beta_transforms_warning()


RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
Resize = register()(T.Resize)
# ToImageTensor = register()(T.ToImageTensor)
# ConvertDtype = register()(T.ConvertDtype)
# PILToTensor = register()(T.PILToTensor)
SanitizeBoundingBoxes = register(name='SanitizeBoundingBoxes')(SanitizeBoundingBoxes)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)

@register()
class RandomZoomOutCond(T.RandomZoomOut):
    def __init__(self, fill: int = 0, side_range=(1.0, 1.9), p: float = 0.5,
                 rare_only: bool = False, common_only: bool = False,
                 rare_classes: Optional[List[int]] = None):
        super().__init__(fill=fill, side_range=side_range)
        self.p = p
        self.rare_only = rare_only
        self.common_only = common_only
        self.rare_classes = set(rare_classes or [])

    def __call__(self, *inputs: Any) -> Any:
        # 확률 체크
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        # target 라벨 확인
        items = inputs if len(inputs) > 1 else inputs[0]
        img, target, *rest = items if isinstance(items, (tuple, list)) else (items, {},)
        labels = set(int(x) for x in target.get("labels", [])) if isinstance(target, dict) else set()
        has_rare = len(labels & self.rare_classes) > 0 if self.rare_classes else False

        # 게이팅 (희귀만/일반만)
        if (self.rare_only and not has_rare) or (self.common_only and has_rare):
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)

@register()
class EmptyTransform(T.Transform):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register()
class PadToSize(T.Pad):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        x = flat_inputs[0]
        if hasattr(F, "get_size"):                 # torchvision >= 0.17
            h, w = F.get_size(x)
        elif hasattr(F, "get_spatial_size"):       # 일부 버전
            h, w = F.get_spatial_size(x)
        else:                                      # 최후 fallback
            if hasattr(x, "shape"):
                h, w = x.shape[-2], x.shape[-1]
            else:  # PIL.Image
                w, h = x.size
        pad_h = max(self.size[1] - h, 0)
        pad_w = max(self.size[0] - w, 0)
        self.padding = [0, 0, pad_w, pad_h]
        return {"padding": self.padding}

    # 2) __init__: fill을 타입-맵으로 넘겨 tv_tensors에도 안전하게 적용되도록
    def __init__(self, size, fill=0, padding_mode='constant') -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        # NOTE: torchvision v2는 fill을 타입별 dict로 받는 걸 지원합니다.
        super().__init__(0, {torch.Tensor: fill, PIL.Image.Image: fill}, padding_mode)

    # 3) tv_tensors.Image 등에도 동작하는 fill 해석기
    def _resolve_fill(self, inpt: Any):
        fm = getattr(self, "_fill", {}) or {}
        t = type(inpt)
        if t in fm:
            return fm[t]
        # tv_tensors.Image/Mask/BoundingBoxes/Video 는 Tensor의 서브클래스
        if isinstance(inpt, torch.Tensor):
            return fm.get(torch.Tensor, 0)
        if isinstance(inpt, PIL.Image.Image):
            return fm.get(PIL.Image.Image, 0)
        return 0

    # 4) _transform: 딱-타입 매칭 대신 안전 해석기 사용
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return F.pad(
            inpt,
            padding=params["padding"],
            fill=self._resolve_fill(inpt),
            padding_mode=self.padding_mode,   # type: ignore[arg-type]
        )


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1,
                 min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2,
                 sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0,
                 rare_only: bool = False, common_only: bool = False, rare_classes: Optional[List[int]] = None):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p
        self.rare_only = rare_only
        self.common_only = common_only
        self.rare_classes = set(rare_classes or [])

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        items = inputs if len(inputs) > 1 else inputs[0]
        img, target, *rest = items if isinstance(items, (tuple, list)) else (items, {},)
        labels = set(int(x) for x in target.get("labels", [])) if isinstance(target, dict) else set()
        has_rare = len(labels & self.rare_classes) > 0 if self.rare_classes else False

        if (self.rare_only and not has_rare) or (self.common_only and has_rare):
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (
        BoundingBoxes,
    )
    def __init__(self, fmt='', normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(inpt, key='boxes', box_format=self.fmt.upper(), spatial_size=spatial_size)

        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt


@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (
        PIL.Image.Image,
    )
    def __init__(self, dtype='float32', scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == 'float32':
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.

        inpt = Image(inpt)

        return inpt
