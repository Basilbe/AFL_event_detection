"""Video decoding and sliding-window utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import torch
from torchvision.io import read_video


@dataclass
class SlidingWindowConfig:
    clip_length: int
    stride: int


def load_video_frames(path: Path | str) -> torch.Tensor:
    """Decode a video file into a ``(T, C, H, W)`` tensor of ``uint8`` frames."""

    video_path = Path(path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file {video_path} does not exist")
    frames, _, _ = read_video(str(video_path), output_format="TCHW")
    return frames  # shape: (T, C, H, W)


def sliding_window(
    frames: torch.Tensor,
    config: SlidingWindowConfig,
    min_total_length: Optional[int] = None,
) -> Iterator[Tuple[torch.Tensor, Tuple[int, int]]]:
    """Generate overlapping clips and their corresponding frame indices."""

    if frames.dim() != 4:
        raise ValueError("Expected frames tensor with shape (T, C, H, W)")

    total_frames = frames.size(0)
    clip_length = config.clip_length
    stride = config.stride

    if min_total_length is not None and total_frames < min_total_length:
        raise ValueError(
            f"Video is shorter ({total_frames}) than required minimum {min_total_length}"
        )

    if total_frames < clip_length:
        padding = clip_length - total_frames
        last_frame = frames[-1:].repeat(padding, 1, 1, 1)
        frames = torch.cat([frames, last_frame], dim=0)
        total_frames = frames.size(0)

    start = 0
    while start + clip_length <= total_frames:
        end = start + clip_length
        clip = frames[start:end].permute(1, 0, 2, 3).float() / 255.0
        yield clip, (start, end)
        start += stride

    if total_frames - clip_length > 0 and (total_frames - clip_length) % stride != 0:
        start = total_frames - clip_length
        end = total_frames
        clip = frames[start:end].permute(1, 0, 2, 3).float() / 255.0
        yield clip, (start, end)
