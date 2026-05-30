
from pathlib import Path

from PIL import Image

from src.module2_distraction.data_loader import (
    DistractionDataConfig,
    create_data_loaders,
)


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), color=(120, 40, 80)).save(path)


def test_distraction_data_loader_splits_imagefolder(tmp_path):
    for class_name in ["safe", "phone"]:
        for index in range(5):
            _write_image(tmp_path / class_name / f"{index}.jpg")

    data = create_data_loaders(
        DistractionDataConfig(
            data_dir=str(tmp_path),
            image_size=32,
            batch_size=2,
            val_split=0.2,
            test_split=0.2,
        )
    )

    assert data.class_names == ["phone", "safe"]
    assert len(data.train_loader.dataset) == 6
    assert len(data.val_loader.dataset) == 2
    assert len(data.test_loader.dataset) == 2
