
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class EvaluationResult:
    metrics: dict[str, Any]
    examples_path: str | None = None
    report_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self) -> EvaluationResult:
        raise NotImplementedError
