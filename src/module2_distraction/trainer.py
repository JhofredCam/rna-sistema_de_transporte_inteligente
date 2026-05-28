
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.module2_distraction.evaluator import evaluate_model


@dataclass
class TrainingConfig:
    epochs: int = 12
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path: str | Path = "models/distraction/best_model.pt"
    model_name: str = "resnet18"
    image_size: int = 224
    extra_metadata: dict[str, Any] = field(default_factory=dict)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += batch_size

    return {
        "loss": running_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


def fit(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    class_names: list[str],
    class_to_idx: dict[str, int],
    config: TrainingConfig,
) -> dict[str, Any]:
    device = torch.device(config.device)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(model, dataloaders["train"], criterion, optimizer, device)
        val_results = evaluate_model(model, dataloaders["val"], class_names, device)
        val_metrics = val_results["metrics"]
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)

        current_f1 = val_metrics["f1_macro"]
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            save_checkpoint(
                model=model,
                checkpoint_path=config.checkpoint_path,
                class_to_idx=class_to_idx,
                class_names=class_names,
                model_name=config.model_name,
                image_size=config.image_size,
                metrics=val_metrics,
                history=history,
                extra_metadata=config.extra_metadata,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break

    model.load_state_dict(best_state)
    checkpoint_dir = Path(config.checkpoint_path).parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (checkpoint_dir / "training_history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    return {"best_epoch": best_epoch, "best_f1_macro": best_f1, "history": history}


def save_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    class_to_idx: dict[str, int],
    class_names: list[str],
    model_name: str,
    image_size: int,
    metrics: dict[str, float],
    history: list[dict[str, Any]],
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx": class_to_idx,
            "idx_to_class": {index: name for index, name in enumerate(class_names)},
            "class_names": class_names,
            "model_name": model_name,
            "image_size": image_size,
            "metrics": metrics,
            "history": history,
            "metadata": extra_metadata or {},
        },
        path,
    )
