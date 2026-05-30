
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def classification_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    labels: list[str] | None = None,
    average: str = "weighted",
) -> dict[str, Any]:
    """Return the core metrics required by the project deliverables."""
    label_indices = list(range(len(labels))) if labels else None
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                labels=label_indices,
                average=average,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                labels=label_indices,
                average=average,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                y_pred,
                labels=label_indices,
                average=average,
                zero_division=0,
            )
        ),
        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                labels=label_indices,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                labels=label_indices,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1_score": float(
            f1_score(
                y_true,
                y_pred,
                labels=label_indices,
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=label_indices,
        ).tolist(),
    }
    metrics["classification_report"] = classification_report(
        y_true,
        y_pred,
        labels=label_indices,
        target_names=labels,
        zero_division=0,
        output_dict=True,
    )
    return metrics
