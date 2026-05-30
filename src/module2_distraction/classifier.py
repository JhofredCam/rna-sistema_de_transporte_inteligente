
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import torch

from .augmentation import build_eval_transforms
from .model import build_model


PREVENTIVE_MEASURES = {
    "phone": "Bloqueo operativo de uso del celular, soporte manos libres solo para emergencias y alertas en cabina.",
    "text": "Politica de cero mensajeria durante la conduccion y auditorias aleatorias de cabina.",
    "sleep": "Pausas activas, control de turnos, deteccion temprana de fatiga y relevos en rutas largas.",
    "drows": "Pausas activas, control de turnos, deteccion temprana de fatiga y relevos en rutas largas.",
    "drink": "Capacitacion sobre hidratacion segura solo en paradas autorizadas.",
    "radio": "Configurar radio/navegacion antes de iniciar la ruta y restringir manipulacion en marcha.",
    "reach": "Asegurar objetos antes de arrancar para evitar alcances hacia asientos traseros.",
    "safe": "Mantener reconocimiento positivo y monitoreo preventivo.",
}


@dataclass
class Prediction:
    image: str
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]


class DriverDistractionClassifier:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        metadata = checkpoint.get("metadata", {})
        self.class_names = metadata["class_names"]
        self.image_size = int(metadata.get("image_size", 224))
        self.model = build_model(
            num_classes=len(self.class_names),
            architecture=metadata.get("architecture", "resnet18"),
            pretrained=False,
            freeze_backbone=False,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.transform = build_eval_transforms(self.image_size)

    def predict(self, image_path: str | Path) -> Prediction:
        image_path = Path(image_path)
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()
            confidence, predicted_idx = probabilities.max(dim=0)

        return Prediction(
            image=str(image_path),
            predicted_label=self.class_names[int(predicted_idx)],
            confidence=float(confidence),
            probabilities={
                class_name: float(probabilities[index])
                for index, class_name in enumerate(self.class_names)
            },
        )

    def predict_many(self, image_paths: list[str | Path]) -> list[Prediction]:
        return [self.predict(path) for path in image_paths]


def preventive_measure_for(label: str) -> str:
    normalized = label.lower()
    for keyword, measure in PREVENTIVE_MEASURES.items():
        if keyword in normalized:
            return measure
    return "Revisar el caso con seguridad vial y reforzar capacitacion del comportamiento observado."
