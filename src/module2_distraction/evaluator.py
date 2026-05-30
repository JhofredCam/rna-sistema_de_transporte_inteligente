
from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from src.shared.base_evaluator import BaseEvaluator, EvaluationResult
from src.shared.metrics import classification_metrics
from src.shared.utils import ensure_dir, save_json


def _dataset_sample_paths(loader: DataLoader) -> list[str | None]:
    dataset = loader.dataset
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        samples = getattr(base_dataset, "samples", [])
        return [samples[index][0] for index in dataset.indices if index < len(samples)]
    samples = getattr(dataset, "samples", [])
    return [sample[0] for sample in samples]


class DistractionEvaluator(BaseEvaluator):
    def __init__(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        class_names: list[str],
        device: torch.device | str,
        output_dir: str | Path,
        max_examples: int = 12,
    ) -> None:
        self.model = model
        self.data_loader = data_loader
        self.class_names = class_names
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.max_examples = max_examples

    def evaluate(self) -> EvaluationResult:
        ensure_dir(self.output_dir)
        self.model.to(self.device)
        self.model.eval()

        y_true: list[int] = []
        y_pred: list[int] = []
        confidences: list[float] = []

        with torch.no_grad():
            for images, labels in self.data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images)
                probabilities = torch.softmax(logits, dim=1)
                confidence, predicted = probabilities.max(dim=1)
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(predicted.cpu().tolist())
                confidences.extend(confidence.cpu().tolist())

        metrics = classification_metrics(y_true, y_pred, labels=self.class_names)
        metrics_path = self.output_dir / "metrics.json"
        save_json(metrics, metrics_path)

        self._save_classification_report_csv(metrics["classification_report"])
        examples_path = self._save_examples(y_true, y_pred, confidences)

        return EvaluationResult(
            metrics=metrics,
            examples_path=str(examples_path),
            report_path=str(metrics_path),
        )

    def _save_classification_report_csv(self, report: dict[str, Any]) -> None:
        rows = []
        for label, values in report.items():
            if isinstance(values, dict):
                rows.append({"class": label, **values})
        path = self.output_dir / "classification_report.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["class", "precision", "recall", "f1-score", "support"],
            )
            writer.writeheader()
            writer.writerows(rows)

    def _save_examples(
        self,
        y_true: list[int],
        y_pred: list[int],
        confidences: list[float],
    ) -> Path:
        paths = _dataset_sample_paths(self.data_loader)
        examples_dir = ensure_dir(self.output_dir / "examples")
        correct_dir = ensure_dir(examples_dir / "correct")
        incorrect_dir = ensure_dir(examples_dir / "incorrect")
        correct_count = 0
        incorrect_count = 0
        examples: list[dict[str, Any]] = []

        for index, (true_idx, pred_idx, confidence) in enumerate(
            zip(y_true, y_pred, confidences)
        ):
            is_correct = true_idx == pred_idx
            if is_correct and correct_count >= self.max_examples:
                continue
            if not is_correct and incorrect_count >= self.max_examples:
                continue

            source_path = Path(paths[index]) if index < len(paths) and paths[index] else None
            destination_path = None
            if source_path is not None and source_path.exists():
                target_dir = correct_dir if is_correct else incorrect_dir
                destination_path = target_dir / f"{index:05d}_{source_path.name}"
                shutil.copy2(source_path, destination_path)

            if is_correct:
                correct_count += 1
            else:
                incorrect_count += 1

            examples.append(
                {
                    "image": str(source_path) if source_path else None,
                    "copied_to": str(destination_path) if destination_path else None,
                    "true_label": self.class_names[true_idx],
                    "predicted_label": self.class_names[pred_idx],
                    "confidence": float(confidence),
                    "correct": is_correct,
                }
            )

            if correct_count >= self.max_examples and incorrect_count >= self.max_examples:
                break

        examples_path = examples_dir / "examples.json"
        save_json({"examples": examples}, examples_path)
        return examples_path
