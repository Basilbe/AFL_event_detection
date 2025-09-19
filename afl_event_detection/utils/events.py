"""Utilities for converting clip-level predictions into discrete events."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch


@dataclass
class EventDetectionConfig:
    min_confidence: float = 0.6
    min_consecutive: int = 2


@dataclass
class DetectedEvent:
    label: str
    start_frame: int
    end_frame: int
    confidence: float


def aggregate_predictions(
    logits: torch.Tensor,
    clip_ranges: Sequence[tuple[int, int]],
    class_names: Sequence[str],
    config: EventDetectionConfig,
) -> Dict[str, List[DetectedEvent]]:
    """Group confident, consecutive predictions into events."""

    probs = torch.softmax(logits, dim=1)
    best_scores, best_indices = probs.max(dim=1)
    events: Dict[str, List[DetectedEvent]] = {name: [] for name in class_names}

    current_label: str | None = None
    current_frames: List[tuple[int, int]] = []
    current_scores: List[float] = []

    for (start, end), score, idx in zip(clip_ranges, best_scores.tolist(), best_indices.tolist()):
        label = class_names[idx]
        if score < config.min_confidence:
            if current_label is not None:
                _maybe_store_event(current_label, current_frames, current_scores, events, config)
                current_label = None
                current_frames = []
                current_scores = []
            continue

        if label == current_label:
            current_frames.append((start, end))
            current_scores.append(score)
        else:
            if current_label is not None:
                _maybe_store_event(current_label, current_frames, current_scores, events, config)
            current_label = label
            current_frames = [(start, end)]
            current_scores = [score]

    if current_label is not None:
        _maybe_store_event(current_label, current_frames, current_scores, events, config)

    return events


def _maybe_store_event(
    label: str,
    frames: List[tuple[int, int]],
    scores: List[float],
    events: Dict[str, List[DetectedEvent]],
    config: EventDetectionConfig,
) -> None:
    if len(frames) < config.min_consecutive:
        return
    start_frame = frames[0][0]
    end_frame = frames[-1][1]
    avg_conf = sum(scores) / len(scores)
    events[label].append(DetectedEvent(label, start_frame, end_frame, avg_conf))
