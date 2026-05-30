
from src.shared.metrics import classification_metrics


def test_classification_metrics_include_required_deliverables():
    metrics = classification_metrics(
        y_true=[0, 1, 1, 2],
        y_pred=[0, 1, 0, 2],
        labels=["safe", "phone", "sleepy"],
    )

    assert metrics["accuracy"] == 0.75
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert len(metrics["confusion_matrix"]) == 3
