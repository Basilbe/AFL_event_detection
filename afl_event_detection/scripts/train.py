"""Command-line entry point for training the AFL action recogniser."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn

from afl_event_detection.data.dataset import ClipSamplingConfig, VideoFolderClips
from afl_event_detection.models.r2plus1d import build_r2plus1d_classifier
from afl_event_detection.training.engine import TrainState, evaluate, save_checkpoint, train_one_epoch
from afl_event_detection.utils import transforms as T


@dataclass
class TrainingConfig:
    data_root: str
    output_dir: str
    epochs: int = 30
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    val_split: float = 0.15
    test_split: float = 0.15
    clip_length: int = 16
    frame_rate: Optional[int] = None
    step_between_clips: int = 8
    resize: Optional[int] = 112
    crop_size: Optional[int] = 112
    dropout: float = 0.5
    use_amp: bool = False
    device: Optional[str] = None


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=str, help="Directory containing class-wise video folders")
    parser.add_argument("output_dir", type=str, help="Where to store checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--frame-rate", type=int, default=None)
    parser.add_argument("--step-between-clips", type=int, default=8)
    parser.add_argument("--resize", type=int, default=112)
    parser.add_argument("--crop-size", type=int, default=112)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision even if CUDA is available")
    parser.add_argument("--device", type=str, default=None, help="Override training device (cpu or cuda)")
    return parser


def main(argv: Optional[Any] = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    config = TrainingConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        val_split=args.val_split,
        test_split=args.test_split,
        clip_length=args.clip_length,
        frame_rate=args.frame_rate,
        step_between_clips=args.step_between_clips,
        resize=args.resize,
        crop_size=args.crop_size,
        dropout=args.dropout,
        use_amp=not args.no_amp,
        device=args.device,
    )

    device = torch.device(
        config.device
        if config.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    sampling = ClipSamplingConfig(
        clip_length=config.clip_length,
        frame_rate=config.frame_rate,
        step_between_clips=config.step_between_clips,
    )

    transform_config: Dict[str, Any] = {
        "resize": config.resize,
        "center_crop": config.crop_size,
        "normalize": True,
        "mean": T.DEFAULT_MEAN,
        "std": T.DEFAULT_STD,
    }
    transform = T.build_transforms(transform_config)

    dataset = VideoFolderClips(config.data_root, sampling, transform=transform)
    train_loader, val_loader, test_loader = dataset.make_dataloaders(
        config.batch_size,
        num_workers=config.num_workers,
        val_split=config.val_split,
        test_split=config.test_split,
    )

    model = build_r2plus1d_classifier(len(dataset.classes), dropout=config.dropout)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    scaler = torch.cuda.amp.GradScaler() if (config.use_amp and device.type == "cuda") else None
    state = TrainState(epoch=0, global_step=0)
    best_path: Optional[Path] = None

    for epoch in range(config.epochs):
        train_stats = train_one_epoch(model, criterion, optimizer, train_loader, device, scaler)
        val_stats = evaluate(model, criterion, val_loader, device)
        scheduler.step()

        state.epoch = epoch + 1
        state.global_step += len(train_loader)

        if val_stats["accuracy"] > state.best_val_accuracy:
            state.best_val_accuracy = val_stats["accuracy"]
            best_path = save_checkpoint(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "epoch": state.epoch,
                    "class_names": dataset.classes,
                    "sampling": asdict(sampling),
                    "transform": transform_config,
                    "training_config": asdict(config),
                    "val_stats": val_stats,
                },
                Path(config.output_dir),
                "best.pt",
            )

        save_checkpoint(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch": state.epoch,
                "class_names": dataset.classes,
                "sampling": asdict(sampling),
                "transform": transform_config,
                "training_config": asdict(config),
                "val_stats": val_stats,
            },
            Path(config.output_dir),
            "last.pt",
        )

        print(
            f"Epoch {epoch + 1}/{config.epochs} - "
            f"train_loss: {train_stats['loss']:.4f} train_acc: {train_stats['accuracy']:.3f} "
            f"val_loss: {val_stats['loss']:.4f} val_acc: {val_stats['accuracy']:.3f}"
        )

    test_stats = evaluate(model, criterion, test_loader, device)
    summary = {
        "best_checkpoint": str(best_path) if best_path else None,
        "final_val_accuracy": state.best_val_accuracy,
        "test_stats": test_stats,
        "class_names": dataset.classes,
    }
    summary_path = Path(config.output_dir) / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Test accuracy: {test_stats['accuracy']:.3f}")


if __name__ == "__main__":
    main()
