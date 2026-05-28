
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.shared.metrics import (
    classification_report_rows,
    confusion_matrix_np,
    multiclass_classification_metrics,
)


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: str | torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    model.to(device)
    all_targets: list[int] = []
    all_predictions: list[int] = []
    all_probabilities: list[np.ndarray] = []

    for images, targets in dataloader:
        images = images.to(device)
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)
        all_targets.extend(targets.cpu().numpy().tolist())
        all_predictions.extend(predictions.cpu().numpy().tolist())
        all_probabilities.extend(probabilities.cpu().numpy())

    return (
        np.asarray(all_targets, dtype=int),
        np.asarray(all_predictions, dtype=int),
        np.asarray(all_probabilities, dtype=float),
    )


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    metrics = multiclass_classification_metrics(y_true, y_pred, len(class_names))
    matrix = confusion_matrix_np(y_true, y_pred, len(class_names))
    report = classification_report_rows(y_true, y_pred, class_names)
    return {
        "metrics": metrics.to_dict(),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "class_names": class_names,
    }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    class_names: list[str],
    device: str | torch.device,
) -> dict[str, Any]:
    y_true, y_pred, _ = collect_predictions(model, dataloader, device)
    return evaluate_predictions(y_true, y_pred, class_names)


def save_evaluation_artifacts(results: dict[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    with (output / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    pd.DataFrame(results["classification_report"]).to_csv(
        output / "classification_report.csv", index=False
    )
    pd.DataFrame(
        results["confusion_matrix"],
        index=results["class_names"],
        columns=results["class_names"],
    ).to_csv(output / "confusion_matrix.csv")


def distraction_frequency_report(
    y_pred: np.ndarray,
    class_names: list[str],
    safe_class_name: str = "Safe Driving",
) -> list[dict[str, float | int | str]]:
    total = len(y_pred)
    rows = []
    for class_index, class_name in enumerate(class_names):
        if class_name == safe_class_name:
            continue
        count = int((y_pred == class_index).sum())
        rows.append(
            {
                "class_name": class_name,
                "predicted_count": count,
                "predicted_share": float(count / total) if total else 0.0,
            }
        )
    return sorted(rows, key=lambda row: row["predicted_count"], reverse=True)
