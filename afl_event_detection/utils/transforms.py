"""Lightweight video transforms operating on ``(C, T, H, W)`` tensors."""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

DEFAULT_MEAN = (0.43216, 0.394666, 0.37645)
DEFAULT_STD = (0.22803, 0.22145, 0.216989)


class Compose:
    def __init__(self, transforms: Sequence[Callable[[torch.Tensor], torch.Tensor]]):
        self.transforms = list(transforms)

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            clip = transform(clip)
        return clip


class Resize:
    def __init__(self, size: int | Tuple[int, int]):
        if isinstance(size, int):
            size = (size, size)
        self.size = size

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        clip_tchw = clip.permute(1, 0, 2, 3)
        resized = F.interpolate(
            clip_tchw,
            size=self.size,
            mode="bilinear",
            align_corners=False,
        )
        return resized.permute(1, 0, 2, 3)


class CenterCrop:
    def __init__(self, size: int | Tuple[int, int]):
        if isinstance(size, int):
            size = (size, size)
        self.size = size

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        _, _, h, w = clip.shape
        th, tw = self.size
        i = max(0, (h - th) // 2)
        j = max(0, (w - tw) // 2)
        return clip[:, :, i : i + th, j : j + tw]


class Normalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.registered_mean = torch.tensor(mean).view(-1, 1, 1, 1)
        self.registered_std = torch.tensor(std).view(-1, 1, 1, 1)

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        return (clip - self.registered_mean) / self.registered_std


def build_transforms(config: Optional[Dict]) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
    if not config:
        return None

    transforms = []
    resize = config.get("resize")
    if resize:
        transforms.append(Resize(resize))
    crop = config.get("center_crop")
    if crop:
        transforms.append(CenterCrop(crop))

    if config.get("normalize", True):
        mean = config.get("mean", DEFAULT_MEAN)
        std = config.get("std", DEFAULT_STD)
        transforms.append(Normalize(mean, std))

    if not transforms:
        return None
    return Compose(transforms)
