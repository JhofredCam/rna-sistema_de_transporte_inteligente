
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .utils import ensure_dir


class BaseTorchModel(nn.Module):
    """Small common wrapper for PyTorch models used by the modules."""

    def save_checkpoint(
        self,
        path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        checkpoint_path = Path(path)
        ensure_dir(checkpoint_path.parent)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "metadata": metadata or {},
            },
            checkpoint_path,
        )

    def load_checkpoint(
        self,
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=map_location)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.load_state_dict(state_dict)
        return checkpoint.get("metadata", {})
