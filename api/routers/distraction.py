
from __future__ import annotations

import tempfile
from pathlib import Path

from dependencies import get_distraction_classifier
from fastapi import APIRouter, File, HTTPException, UploadFile

from src.module2_distraction.classifier import preventive_measure_for


router = APIRouter(prefix="/distraction", tags=["Clasificacion de Distraccion"])


@router.post("/predict")
async def predict_distraction(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen.")

    classifier = get_distraction_classifier()
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo de distraccion no disponible. Entrene el modulo 2 y copie el checkpoint.",
        )

    suffix = Path(file.filename or "image.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(await file.read())
        temp_path = Path(temp_file.name)

    try:
        prediction = classifier.predict(temp_path)
        return {
            "predicted_label": prediction.predicted_label,
            "confidence": prediction.confidence,
            "probabilities": prediction.probabilities,
            "preventive_measure": preventive_measure_for(prediction.predicted_label),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)
