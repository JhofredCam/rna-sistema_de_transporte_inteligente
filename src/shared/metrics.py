
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision_macro: float
    precision_weighted: float
    recall_macro: float
    recall_weighted: float
    f1_macro: float
    f1_weighted: float

    def to_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision_macro": self.precision_macro,
            "precision_weighted": self.precision_weighted,
            "recall_macro": self.recall_macro,
            "recall_weighted": self.recall_weighted,
            "f1_macro": self.f1_macro,
            "f1_weighted": self.f1_weighted,
        }


def _as_numpy(values: Iterable[int] | np.ndarray) -> np.ndarray:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values)
    if array.ndim != 1:
        raise ValueError("Expected a one-dimensional array of labels.")
    return array.astype(int)


def confusion_matrix_np(
    y_true: Iterable[int] | np.ndarray,
    y_pred: Iterable[int] | np.ndarray,
    num_classes: int,
) -> np.ndarray:
    true = _as_numpy(y_true)
    pred = _as_numpy(y_pred)
    if true.shape[0] != pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for expected, predicted in zip(true, pred):
        if 0 <= expected < num_classes and 0 <= predicted < num_classes:
            matrix[expected, predicted] += 1
    return matrix


def precision_recall_f1_from_confusion(matrix: np.ndarray) -> dict[str, np.ndarray]:
    matrix = np.asarray(matrix)
    true_positive = np.diag(matrix).astype(float)
    predicted_positive = matrix.sum(axis=0).astype(float)
    actual_positive = matrix.sum(axis=1).astype(float)

    precision = np.divide(
        true_positive,
        predicted_positive,
        out=np.zeros_like(true_positive),
        where=predicted_positive != 0,
    )
    recall = np.divide(
        true_positive,
        actual_positive,
        out=np.zeros_like(true_positive),
        where=actual_positive != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    return {"precision": precision, "recall": recall, "f1": f1, "support": actual_positive}


def multiclass_classification_metrics(
    y_true: Iterable[int] | np.ndarray,
    y_pred: Iterable[int] | np.ndarray,
    num_classes: int,
) -> ClassificationMetrics:
    true = _as_numpy(y_true)
    pred = _as_numpy(y_pred)
    if true.shape[0] == 0:
        raise ValueError("Cannot compute metrics on an empty label array.")

    matrix = confusion_matrix_np(true, pred, num_classes)
    per_class = precision_recall_f1_from_confusion(matrix)
    support = per_class["support"]
    total = support.sum()
    weights = np.divide(support, total, out=np.zeros_like(support), where=total != 0)

    accuracy = float((true == pred).sum() / true.shape[0])
    return ClassificationMetrics(
        accuracy=accuracy,
        precision_macro=float(per_class["precision"].mean()),
        precision_weighted=float((per_class["precision"] * weights).sum()),
        recall_macro=float(per_class["recall"].mean()),
        recall_weighted=float((per_class["recall"] * weights).sum()),
        f1_macro=float(per_class["f1"].mean()),
        f1_weighted=float((per_class["f1"] * weights).sum()),
    )


def classification_report_rows(
    y_true: Iterable[int] | np.ndarray,
    y_pred: Iterable[int] | np.ndarray,
    class_names: list[str],
) -> list[dict[str, float | int | str]]:
    matrix = confusion_matrix_np(y_true, y_pred, len(class_names))
    per_class = precision_recall_f1_from_confusion(matrix)
    rows = []
    for index, class_name in enumerate(class_names):
        rows.append(
            {
                "class_name": class_name,
                "precision": float(per_class["precision"][index]),
                "recall": float(per_class["recall"][index]),
                "f1_score": float(per_class["f1"][index]),
                "support": int(per_class["support"][index]),
            }
        )
    return rows
