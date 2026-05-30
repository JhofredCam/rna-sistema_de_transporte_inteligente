
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    tqdm = None

from src.shared.base_trainer import BaseTrainer, TrainingResult
from src.shared.metrics import classification_metrics
from src.shared.utils import ensure_dir, get_device, save_json, set_seed

from .data_loader import DistractionDataConfig, create_data_loaders
from .evaluator import DistractionEvaluator
from .model import build_model


@dataclass
class DistractionTrainingConfig:
    data_dir: str
    output_dir: str = "models/module2_distraction"
    architecture: str = "resnet18"
    pretrained: bool = True
    freeze_backbone: bool = False
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 4
    num_workers: int = 0
    seed: int = 42
    val_split: float = 0.15
    test_split: float = 0.15
    device: str | None = None


class DistractionTrainer(BaseTrainer):
    def __init__(self, config: DistractionTrainingConfig) -> None:
        super().__init__(config.output_dir)
        self.config = config
        self.device = torch.device(config.device) if config.device else get_device()

    def train(self) -> TrainingResult:
        set_seed(self.config.seed)
        ensure_dir(self.output_dir)

        data = create_data_loaders(
            DistractionDataConfig(
                data_dir=self.config.data_dir,
                image_size=self.config.image_size,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                val_split=self.config.val_split,
                test_split=self.config.test_split,
                seed=self.config.seed,
            )
        )

        model = build_model(
            num_classes=len(data.class_names),
            architecture=self.config.architecture,
            pretrained=self.config.pretrained,
            dropout=0.3,
            freeze_backbone=self.config.freeze_backbone,
        ).to(self.device)

        optimizer = AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

        best_f1 = -1.0
        epochs_without_improvement = 0
        history: list[dict[str, float]] = []
        best_model_path = self.output_dir / "best_model.pth"

        for epoch in range(1, self.config.epochs + 1):
            print(f"Epoch {epoch}/{self.config.epochs} - entrenando...")
            train_loss = self._train_one_epoch(
                model,
                data.train_loader,
                criterion,
                optimizer,
                epoch,
            )
            val_loss, y_true, y_pred = self._predict_epoch(
                model,
                data.val_loader,
                criterion,
                epoch,
            )
            val_metrics = classification_metrics(y_true, y_pred, labels=data.class_names)
            val_f1 = val_metrics["f1_score"]
            scheduler.step(val_f1)

            row = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1_score": val_f1,
            }
            history.append(row)
            self._save_history(history)
            print(
                "Epoch "
                f"{epoch}/{self.config.epochs} - "
                f"loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_f1:.4f}"
            )

            if val_f1 > best_f1:
                best_f1 = val_f1
                epochs_without_improvement = 0
                model.save_checkpoint(
                    best_model_path,
                    metadata={
                        "class_names": data.class_names,
                        "class_to_idx": data.class_to_idx,
                        "architecture": self.config.architecture,
                        "image_size": self.config.image_size,
                        "pretrained": self.config.pretrained,
                        "freeze_backbone": self.config.freeze_backbone,
                        "best_val_f1_score": best_f1,
                    },
                )
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.config.patience:
                break

        metadata = {
            "config": asdict(self.config),
            "class_names": data.class_names,
            "class_to_idx": data.class_to_idx,
            "best_val_f1_score": best_f1,
            "device": str(self.device),
            "history": history,
        }
        save_json(metadata, self.output_dir / "metadata.json")

        checkpoint = torch.load(best_model_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        evaluator = DistractionEvaluator(
            model=model,
            data_loader=data.test_loader,
            class_names=data.class_names,
            device=self.device,
            output_dir=self.output_dir / "evaluation",
        )
        evaluation = evaluator.evaluate()

        return TrainingResult(
            best_model_path=str(best_model_path),
            metrics_path=evaluation.report_path,
            history=history,
            metadata=metadata,
        )

    def _train_one_epoch(
        self,
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> float:
        model.train()
        total_loss = 0.0
        total_samples = 0
        progress = self._progress_bar(loader, f"Epoch {epoch}/{self.config.epochs} train")

        for images, labels in progress:
            images = images.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            if tqdm is not None:
                progress.set_postfix(loss=f"{total_loss / max(total_samples, 1):.4f}")

        return total_loss / max(total_samples, 1)

    def _predict_epoch(
        self,
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        epoch: int,
    ) -> tuple[float, list[int], list[int]]:
        model.eval()
        total_loss = 0.0
        total_samples = 0
        y_true: list[int] = []
        y_pred: list[int] = []
        progress = self._progress_bar(loader, f"Epoch {epoch}/{self.config.epochs} val")

        with torch.no_grad():
            for images, labels in progress:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = model(images)
                loss = criterion(logits, labels)
                predicted = logits.argmax(dim=1)
                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(predicted.cpu().tolist())
                if tqdm is not None:
                    progress.set_postfix(loss=f"{total_loss / max(total_samples, 1):.4f}")

        return total_loss / max(total_samples, 1), y_true, y_pred

    def _save_history(self, history: list[dict[str, float]]) -> None:
        path = self.output_dir / "history.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    def _progress_bar(
        self,
        loader: torch.utils.data.DataLoader,
        description: str,
    ):
        if tqdm is None:
            return loader
        return tqdm(
            loader,
            desc=description,
            total=len(loader),
            unit="batch",
            leave=False,
        )


def train_distraction_model(config: DistractionTrainingConfig) -> TrainingResult:
    return DistractionTrainer(config).train()
