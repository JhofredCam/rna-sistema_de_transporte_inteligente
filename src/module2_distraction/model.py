
from __future__ import annotations

from typing import Literal

import torch.nn as nn
from torchvision import models


SupportedBackbone = Literal["resnet18", "resnet34", "mobilenet_v3_small"]


def build_driver_behavior_model(
    num_classes: int,
    model_name: SupportedBackbone = "resnet18",
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than 1.")

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        feature_modules = [name for name, _ in model.named_parameters() if not name.startswith("fc.")]
    elif model_name == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        feature_modules = [name for name, _ in model.named_parameters() if not name.startswith("fc.")]
    elif model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        feature_modules = [
            name for name, _ in model.named_parameters() if not name.startswith("classifier.")
        ]
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    if freeze_backbone:
        for name, parameter in model.named_parameters():
            if name in feature_modules:
                parameter.requires_grad = False
    return model
