"""Command-line inference script for AFL action recognition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from afl_event_detection.models.r2plus1d import build_r2plus1d_classifier
from afl_event_detection.utils.events import EventDetectionConfig, aggregate_predictions
from afl_event_detection.utils.transforms import build_transforms
from afl_event_detection.utils.video import SlidingWindowConfig, load_video_frames, sliding_window


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=str, help="Path to the trained model checkpoint")
    parser.add_argument("video", type=str, help="Video file to analyse")
    parser.add_argument("--window-stride", type=int, default=8, help="Stride between consecutive clips")
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument("--min-consecutive", type=int, default=2)
    parser.add_argument("--output-json", type=str, default=None, help="Optional file to store predictions")
    parser.add_argument("--device", type=str, default=None)
    return parser


def main(argv: Optional[Any] = None) -> None:
    args = build_argparser().parse_args(argv)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_names: List[str] = checkpoint["class_names"]
    sampling_cfg = checkpoint["sampling"]
    transform_cfg = checkpoint.get("transform", {})

    model = build_r2plus1d_classifier(len(class_names))
    model.load_state_dict(checkpoint["model_state"])

    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device)
    model.eval()

    transform = build_transforms(transform_cfg)
    frames = load_video_frames(args.video)

    window_cfg = SlidingWindowConfig(
        clip_length=sampling_cfg["clip_length"],
        stride=args.window_stride,
    )

    clips = []
    ranges = []
    for clip, (start, end) in sliding_window(frames, window_cfg):
        if transform is not None:
            clip = transform(clip)
        clips.append(clip)
        ranges.append((start, end))

    if not clips:
        raise RuntimeError("No clips generated from the provided video")

    batch = torch.stack(clips).to(device)
    with torch.no_grad():
        logits = model(batch)

    events = aggregate_predictions(
        logits.cpu(),
        ranges,
        class_names,
        EventDetectionConfig(
            min_confidence=args.min_confidence,
            min_consecutive=args.min_consecutive,
        ),
    )

    summary: Dict[str, Any] = {
        "video": str(Path(args.video).resolve()),
        "counts": {label: len(event_list) for label, event_list in events.items()},
        "events": {
            label: [
                {
                    "start_frame": event.start_frame,
                    "end_frame": event.end_frame,
                    "confidence": event.confidence,
                }
                for event in event_list
            ]
            for label, event_list in events.items()
        },
    }

    print(json.dumps(summary, indent=2))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
