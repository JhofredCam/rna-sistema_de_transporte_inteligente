
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import ImageFolder

from .augmentation import build_eval_transforms, build_train_transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRAIN_DIR_NAMES = ("train", "training", "Train", "Training")
VAL_DIR_NAMES = ("val", "valid", "validation", "Val", "Validation")
TEST_DIR_NAMES = ("test", "testing", "Test", "Testing")


@dataclass
class DistractionDataConfig:
    data_dir: str
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 0
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42


@dataclass
class DistractionDataModule:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]
    class_to_idx: dict[str, int]


def _find_child(root: Path, candidates: tuple[str, ...]) -> Path | None:
    for name in candidates:
        child = root / name
        if child.exists() and child.is_dir():
            return child
    return None


def _contains_images(path: Path) -> bool:
    return any(
        child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
        for child in path.iterdir()
    )


def _looks_like_imagefolder_root(path: Path) -> bool:
    children = [child for child in path.iterdir() if child.is_dir()]
    if not children:
        return False
    if _find_child(path, TRAIN_DIR_NAMES + VAL_DIR_NAMES + TEST_DIR_NAMES) is not None:
        return True
    return sum(_contains_images(child) for child in children) >= 2


def resolve_dataset_root(root: Path) -> Path:
    """Skip common one-folder wrappers produced by Kaggle zip extraction."""
    current = root
    while not _looks_like_imagefolder_root(current):
        children = [child for child in current.iterdir() if child.is_dir()]
        files = [child for child in current.iterdir() if child.is_file()]
        if len(children) != 1 or files:
            break
        current = children[0]
    return current


def _imagefolder(root: Path, image_size: int, train: bool) -> ImageFolder:
    transform = build_train_transforms(image_size) if train else build_eval_transforms(image_size)
    return ImageFolder(root=str(root), transform=transform)


def create_data_loaders(config: DistractionDataConfig) -> DistractionDataModule:
    root = Path(config.data_dir)
    if not root.exists():
        raise FileNotFoundError(f"No existe el directorio del dataset: {root}")
    root = resolve_dataset_root(root)

    train_dir = _find_child(root, TRAIN_DIR_NAMES)
    val_dir = _find_child(root, VAL_DIR_NAMES)
    test_dir = _find_child(root, TEST_DIR_NAMES)

    generator = torch.Generator().manual_seed(config.seed)

    if train_dir is not None:
        train_dataset = _imagefolder(train_dir, config.image_size, train=True)
        val_dataset = (
            _imagefolder(val_dir, config.image_size, train=False)
            if val_dir is not None
            else None
        )
        test_dataset = (
            _imagefolder(test_dir, config.image_size, train=False)
            if test_dir is not None
            else None
        )
        class_names = train_dataset.classes
        class_to_idx = train_dataset.class_to_idx

        if val_dataset is None or test_dataset is None:
            eval_dataset = _imagefolder(train_dir, config.image_size, train=False)
            total = len(eval_dataset)
            val_size = int(total * config.val_split) if val_dataset is None else 0
            test_size = int(total * config.test_split) if test_dataset is None else 0
            train_size = total - val_size - test_size
            if train_size <= 0:
                raise ValueError("Los splits dejan el conjunto de entrenamiento sin datos.")
            split_lengths = [train_size]
            if val_dataset is None:
                split_lengths.append(val_size)
            if test_dataset is None:
                split_lengths.append(test_size)
            splits = random_split(
                eval_dataset,
                split_lengths,
                generator=generator,
            )
            train_eval = splits[0]
            split_position = 1
            train_dataset = Subset(_imagefolder(train_dir, config.image_size, train=True), train_eval.indices)
            if val_dataset is None:
                val_dataset = splits[split_position]
                split_position += 1
            if test_dataset is None:
                test_dataset = splits[split_position]
    else:
        train_base = _imagefolder(root, config.image_size, train=True)
        eval_base = _imagefolder(root, config.image_size, train=False)
        total = len(train_base)
        val_size = int(total * config.val_split)
        test_size = int(total * config.test_split)
        train_size = total - val_size - test_size
        if train_size <= 0 or val_size <= 0 or test_size <= 0:
            raise ValueError(
                "Dataset insuficiente para crear train/val/test. Ajuste val_split/test_split."
            )
        train_indices, val_indices, test_indices = random_split(
            range(total),
            [train_size, val_size, test_size],
            generator=generator,
        )
        train_dataset = Subset(train_base, list(train_indices))
        val_dataset = Subset(eval_base, list(val_indices))
        test_dataset = Subset(eval_base, list(test_indices))
        class_names = train_base.classes
        class_to_idx = train_base.class_to_idx

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return DistractionDataModule(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=class_names,
        class_to_idx=class_to_idx,
    )
