"""Data loading utilities for AFL action recognition."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets.video_utils import VideoClips


@dataclass
class ClipSamplingConfig:
    """Configuration for sampling fixed-length clips from gameplay footage."""

    clip_length: int = 16
    frame_rate: Optional[int] = None
    step_between_clips: int = 8


class VideoFolderClips(Dataset):
    """Dataset that samples short clips from videos grouped by class folders.

    The directory structure is expected to follow the pattern::

        root/
            action_0/
                clip_000.mp4
                clip_001.mp4
            action_1/
                clip_101.mp4

    Each video is segmented into evenly spaced clips according to
    :class:`ClipSamplingConfig`.  By default the dataset returns tensors with the
    shape ``(C, T, H, W)`` in the ``float32`` range ``[0, 1]``.
    """

    def __init__(
        self,
        root: Path | str,
        sampling: ClipSamplingConfig,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root {self.root} does not exist")

        self.transform = transform
        self.classes, video_paths, self.class_to_idx = self._discover_dataset()
        self.video_clips = VideoClips(
            video_paths,
            sampling.clip_length,
            sampling.step_between_clips,
            frame_rate=sampling.frame_rate,
        )

    def _discover_dataset(self) -> Tuple[List[str], List[str], Dict[str, int]]:
        classes = sorted(entry.name for entry in self.root.iterdir() if entry.is_dir())
        if not classes:
            raise RuntimeError(f"No class subdirectories found under {self.root}")

        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        video_paths: List[str] = []
        for cls in classes:
            for video in sorted((self.root / cls).glob("*")):
                if video.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv"}:
                    continue
                video_paths.append(str(video.resolve()))
        if not video_paths:
            raise RuntimeError(f"No supported video files found in {self.root}")
        return classes, video_paths, class_to_idx

    def __len__(self) -> int:  # type: ignore[override]
        return self.video_clips.num_clips()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:  # type: ignore[override]
        clip, _, _ = self.video_clips.get_clip(index)
        video_idx = self.video_clips.clips_metadata["video_idx"][index]
        label = self.class_to_idx[self._class_for_video(video_idx)]
        clip = clip.permute(3, 0, 1, 2).float() / 255.0  # (T, H, W, C) -> (C, T, H, W)
        if self.transform:
            clip = self.transform(clip)
        return clip, label

    def _class_for_video(self, video_index: int) -> str:
        video_path = Path(self.video_clips.video_paths[video_index])
        return video_path.parent.name

    def make_dataloaders(
        self,
        batch_size: int,
        num_workers: int = 4,
        val_split: float = 0.15,
        test_split: float = 0.15,
        seed: int = 42,
        drop_last: bool = False,
        persistent_workers: bool = False,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train/validation/test dataloaders with random stratified splits."""

        if val_split < 0 or test_split < 0 or val_split + test_split >= 1.0:
            raise ValueError("Validation and test splits must sum to < 1.0")

        total_len = len(self)
        val_len = int(total_len * val_split)
        test_len = int(total_len * test_split)
        train_len = total_len - val_len - test_len
        generator = torch.Generator().manual_seed(seed)
        subsets = random_split(self, [train_len, val_len, test_len], generator=generator)

        def _loader(subset: Dataset[Tuple[torch.Tensor, int]], shuffle: bool) -> DataLoader:
            return DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=drop_last and shuffle,
                persistent_workers=persistent_workers and num_workers > 0,
            )

        train_loader = _loader(subsets[0], shuffle=True)
        val_loader = _loader(subsets[1], shuffle=False)
        test_loader = _loader(subsets[2], shuffle=False)
        return train_loader, val_loader, test_loader


def collate_clips(batch: Sequence[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate a batch of clips with padding-aware stacking."""

    clips, labels = zip(*batch)
    clip_tensor = torch.stack(clips, dim=0)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return clip_tensor, label_tensor
