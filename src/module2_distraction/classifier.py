
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from src.module2_distraction.augmentation import build_eval_transforms
from src.module2_distraction.model import build_driver_behavior_model


@dataclass(frozen=True)
class Prediction:
    image_path: str
    predicted_class: str
    predicted_index: int
    confidence: float
    probabilities: dict[str, float]


class DriverDistractionClassifier:
    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.class_names = list(checkpoint["class_names"])
        self.image_size = int(checkpoint.get("image_size", 224))
        self.model_name = checkpoint.get("model_name", "resnet18")
        self.model = build_driver_behavior_model(
            num_classes=len(self.class_names),
            model_name=self.model_name,
            pretrained=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.transform = build_eval_transforms(self.image_size)

    @torch.no_grad()
    def predict_image(self, image_path: str | Path) -> Prediction:
        path = Path(image_path)
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probabilities_tensor = torch.softmax(logits, dim=1).squeeze(0).cpu()
        predicted_index = int(probabilities_tensor.argmax().item())
        probabilities = {
            class_name: float(probabilities_tensor[index].item())
            for index, class_name in enumerate(self.class_names)
        }
        return Prediction(
            image_path=str(path),
            predicted_class=self.class_names[predicted_index],
            predicted_index=predicted_index,
            confidence=probabilities[self.class_names[predicted_index]],
            probabilities=probabilities,
        )

    def predict_folder(self, folder_path: str | Path) -> list[Prediction]:
        folder = Path(folder_path)
        image_paths = sorted(
            path
            for path in folder.rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        return [self.predict_image(path) for path in image_paths]
