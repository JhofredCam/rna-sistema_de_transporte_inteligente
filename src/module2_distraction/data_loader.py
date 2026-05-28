
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from src.config import DRIVER_BEHAVIOR_CLASSES
from src.module2_distraction.augmentation import build_eval_transforms, build_train_transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class DatasetSplits:
    train: Subset
    val: Subset
    test: Subset
    class_to_idx: dict[str, int]
    idx_to_class: dict[int, str]


def find_imagefolder_root(data_root: str | Path) -> Path:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    if _looks_like_imagefolder(root):
        return root

    candidates = [path for path in root.rglob("*") if path.is_dir() and _looks_like_imagefolder(path)]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        expected = {name.lower() for name in DRIVER_BEHAVIOR_CLASSES}
        exact = [
            path
            for path in candidates
            if expected.issubset({child.name.lower() for child in path.iterdir() if child.is_dir()})
        ]
        if exact:
            return exact[0]
        return sorted(candidates, key=lambda path: len(path.parts))[0]
    raise ValueError(
        "Could not find an ImageFolder-compatible directory. Expected class subfolders "
        f"under {root}."
    )


def _looks_like_imagefolder(path: Path) -> bool:
    class_dirs = [child for child in path.iterdir() if child.is_dir()]
    if len(class_dirs) < 2:
        return False
    return any(
        file.suffix.lower() in IMAGE_EXTENSIONS
        for class_dir in class_dirs
        for file in class_dir.iterdir()
        if file.is_file()
    )


def validate_expected_classes(class_to_idx: dict[str, int], strict: bool = False) -> None:
    expected = set(DRIVER_BEHAVIOR_CLASSES)
    found = set(class_to_idx)
    missing = expected - found
    if strict and missing:
        raise ValueError(f"Missing expected classes: {sorted(missing)}")
    if len(found) < 2:
        raise ValueError("At least two classes are required for classification.")


def create_dataset_splits(
    data_root: str | Path,
    image_size: int = 224,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    strict_classes: bool = False,
) -> DatasetSplits:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to less than 1.")

    folder_root = find_imagefolder_root(data_root)
    base_dataset = ImageFolder(folder_root)
    validate_expected_classes(base_dataset.class_to_idx, strict=strict_classes)

    train_dataset = ImageFolder(folder_root, transform=build_train_transforms(image_size))
    eval_dataset = ImageFolder(folder_root, transform=build_eval_transforms(image_size))

    indices_by_class: dict[int, list[int]] = {}
    for index, (_, class_index) in enumerate(base_dataset.samples):
        indices_by_class.setdefault(class_index, []).append(index)

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    rng = random.Random(seed)

    for indices in indices_by_class.values():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        total = len(shuffled)
        test_count = max(1, int(round(total * test_ratio))) if total >= 3 and test_ratio > 0 else 0
        val_count = max(1, int(round(total * val_ratio))) if total >= 3 and val_ratio > 0 else 0
        if test_count + val_count >= total:
            test_count = 1 if total >= 3 and test_ratio > 0 else 0
            val_count = 1 if total - test_count >= 2 and val_ratio > 0 else 0
        test_indices.extend(shuffled[:test_count])
        val_indices.extend(shuffled[test_count : test_count + val_count])
        train_indices.extend(shuffled[test_count + val_count :])

    idx_to_class = {index: class_name for class_name, index in base_dataset.class_to_idx.items()}
    return DatasetSplits(
        train=Subset(train_dataset, train_indices),
        val=Subset(eval_dataset, val_indices),
        test=Subset(eval_dataset, test_indices),
        class_to_idx=base_dataset.class_to_idx,
        idx_to_class=idx_to_class,
    )


def create_dataloaders(
    data_root: str | Path,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    strict_classes: bool = False,
) -> tuple[dict[str, DataLoader], DatasetSplits]:
    splits = create_dataset_splits(
        data_root=data_root,
        image_size=image_size,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        strict_classes=strict_classes,
    )
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(
            splits.train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=generator,
        ),
        "val": DataLoader(
            splits.val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            splits.test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }
    return loaders, splits
