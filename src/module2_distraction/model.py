
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision import models

from src.shared.base_model import BaseTorchModel


@dataclass
class DistractionModelConfig:
    num_classes: int
    architecture: str = "resnet18"
    pretrained: bool = True
    dropout: float = 0.3
    freeze_backbone: bool = False


class DriverDistractionModel(BaseTorchModel):
    """Transfer-learning image classifier for distracted driving behaviors."""

    def __init__(self, config: DistractionModelConfig) -> None:
        super().__init__()
        self.config = config
        self.network = self._build_network(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def _build_network(self, config: DistractionModelConfig) -> nn.Module:
        architecture = config.architecture.lower()
        weights = "DEFAULT" if config.pretrained else None

        if architecture == "resnet18":
            network = models.resnet18(weights=weights)
            in_features = network.fc.in_features
            if config.freeze_backbone:
                for parameter in network.parameters():
                    parameter.requires_grad = False
            network.fc = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(in_features, config.num_classes),
            )
            return network

        if architecture == "mobilenet_v3_small":
            network = models.mobilenet_v3_small(weights=weights)
            in_features = network.classifier[-1].in_features
            if config.freeze_backbone:
                for parameter in network.features.parameters():
                    parameter.requires_grad = False
            network.classifier[-1] = nn.Linear(in_features, config.num_classes)
            return network

        if architecture == "efficientnet_b0":
            network = models.efficientnet_b0(weights=weights)
            in_features = network.classifier[-1].in_features
            if config.freeze_backbone:
                for parameter in network.features.parameters():
                    parameter.requires_grad = False
            network.classifier = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(in_features, config.num_classes),
            )
            return network

        raise ValueError(
            "Arquitectura no soportada. Use resnet18, mobilenet_v3_small o efficientnet_b0."
        )


def build_model(
    num_classes: int,
    architecture: str = "resnet18",
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
) -> DriverDistractionModel:
    return DriverDistractionModel(
        DistractionModelConfig(
            num_classes=num_classes,
            architecture=architecture,
            pretrained=pretrained,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        )
    )
