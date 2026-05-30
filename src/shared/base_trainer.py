
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainingResult:
    best_model_path: str
    metrics_path: str | None
    history: list[dict[str, float]]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseTrainer(ABC):
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)

    @abstractmethod
    def train(self) -> TrainingResult:
        raise NotImplementedError
