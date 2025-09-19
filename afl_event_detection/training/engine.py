"""Training and evaluation loops for AFL event detection."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainState:
    epoch: int
    global_step: int
    best_val_accuracy: float = 0.0


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0

    for clips, labels in dataloader:
        clips = clips.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(clips)
            loss = criterion(outputs, labels)

        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        batch_size = clips.size(0)
        running_loss += loss.item() * batch_size
        running_acc += accuracy_from_logits(outputs.detach(), labels) * batch_size
        total_samples += batch_size

    return {
        "loss": running_loss / total_samples,
        "accuracy": running_acc / total_samples,
    }


def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0

    with torch.no_grad():
        for clips, labels in dataloader:
            clips = clips.to(device)
            labels = labels.to(device)
            outputs = model(clips)
            loss = criterion(outputs, labels)

            batch_size = clips.size(0)
            running_loss += loss.item() * batch_size
            running_acc += accuracy_from_logits(outputs, labels) * batch_size
            total_samples += batch_size

    return {
        "loss": running_loss / total_samples,
        "accuracy": running_acc / total_samples,
    }


def save_checkpoint(
    state: Dict[str, torch.Tensor | int | float | Dict],
    output_dir: Path,
    filename: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / filename
    torch.save(state, checkpoint_path)
    return checkpoint_path
