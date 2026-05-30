
import torch

from src.module2_distraction.model import build_model


def test_driver_distraction_model_forward_shape():
    model = build_model(num_classes=4, architecture="resnet18", pretrained=False)
    model.eval()

    with torch.no_grad():
        output = model(torch.randn(2, 3, 224, 224))

    assert output.shape == (2, 4)
