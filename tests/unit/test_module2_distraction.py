from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from PIL import Image

from src.module2_distraction.classifier import DriverDistractionClassifier
from src.module2_distraction.data_loader import create_dataset_splits
from src.module2_distraction.model import build_driver_behavior_model
from src.shared.metrics import multiclass_classification_metrics


def _write_image(path, color):
    image = Image.new("RGB", (32, 32), color=color)
    image.save(path)


def test_dataset_splits_are_reproducible(tmp_path):
    for class_name, color in {"Safe Driving": "green", "Texting Phone": "red"}.items():
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for index in range(6):
            _write_image(class_dir / f"{index}.jpg", color)

    first = create_dataset_splits(tmp_path, image_size=32, seed=123)
    second = create_dataset_splits(tmp_path, image_size=32, seed=123)

    assert first.class_to_idx == {"Safe Driving": 0, "Texting Phone": 1}
    assert first.train.indices == second.train.indices
    assert first.val.indices == second.val.indices
    assert first.test.indices == second.test.indices
    assert len(first.train) + len(first.val) + len(first.test) == 12


def test_model_outputs_expected_logits_shape():
    model = build_driver_behavior_model(num_classes=5, pretrained=False)
    batch = torch.randn(2, 3, 224, 224)

    logits = model(batch)

    assert tuple(logits.shape) == (2, 5)


def test_multiclass_metrics_known_values():
    metrics = multiclass_classification_metrics(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 1, 1],
        num_classes=2,
    )

    assert metrics.accuracy == pytest.approx(0.75)
    assert metrics.recall_macro == pytest.approx(0.75)
    assert metrics.precision_macro == pytest.approx((1.0 + 2 / 3) / 2)


def test_classifier_loads_checkpoint_and_predicts(tmp_path):
    class_names = ["Safe Driving", "Texting Phone"]
    model = build_driver_behavior_model(num_classes=2, pretrained=False)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx": {"Safe Driving": 0, "Texting Phone": 1},
            "idx_to_class": {0: "Safe Driving", 1: "Texting Phone"},
            "class_names": class_names,
            "model_name": "resnet18",
            "image_size": 32,
            "metrics": {},
        },
        checkpoint_path,
    )
    image_path = tmp_path / "driver.jpg"
    _write_image(image_path, "blue")

    classifier = DriverDistractionClassifier(checkpoint_path, device="cpu")
    prediction = classifier.predict_image(image_path)

    assert prediction.predicted_class in class_names
    assert set(prediction.probabilities) == set(class_names)
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0)
