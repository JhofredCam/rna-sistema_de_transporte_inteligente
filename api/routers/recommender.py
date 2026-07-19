from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import ROOT_DIR, DEVICE

router = APIRouter(prefix="/recommender", tags=["Recomendación de Destinos"])

_recommender = None


def _get_recommender():
    global _recommender
    if _recommender is None:
        try:
            from src.module3_recommender.recommender import TravelDestinationRecommender

            checkpoint_path = ROOT_DIR / "models" / "module3_recommender" / "best_model.pth"
            if not checkpoint_path.exists():
                return None
            _recommender = TravelDestinationRecommender(
                checkpoint_path=checkpoint_path,
                device=DEVICE,
            )
            print(f"Modelo recomendador cargado en {DEVICE}: {checkpoint_path}")
        except Exception as e:
            print(f"Error cargando modelo recomendador: {e}")
            return None
    return _recommender


class PreferenceRequest(BaseModel):
    trip_type: str | None = Field(default=None, description="Tipo de viaje preferido")
    budget: str | None = Field(default=None, description="Presupuesto: bajo, medio, alto")
    interests: list[str] | None = Field(default=None, description="Lista de intereses del viajero")
    user_id: str | None = Field(default=None, description="ID de usuario existente (opcional)")
    top_k: int = Field(default=5, ge=1, le=20, description="Número de recomendaciones")


def _content_based_recommend(
    rec,
    preferences: dict[str, Any],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    keywords = set()
    if preferences.get("trip_type"):
        keywords.add(preferences["trip_type"].lower())
    if preferences.get("interests"):
        keywords.update(i.lower() for i in preferences["interests"])

    item_scores = []
    for item_idx in range(len(rec.idx_to_item)):
        metadata = rec.item_metadata[item_idx] if item_idx < len(rec.item_metadata) else {}
        name = rec._display_name(item_idx)

        meta_str = " ".join(str(v) for v in metadata.values()).lower()
        name_lower = str(name).lower()

        score = 0.0
        for kw in keywords:
            if kw in meta_str or kw in name_lower:
                score += 1.0

        if preferences.get("budget") == "bajo":
            for cheap_kw in ["budget", "cheap", "affordable", "hostel", "backpacker"]:
                if cheap_kw in meta_str:
                    score += 0.5
                    break
        elif preferences.get("budget") == "alto":
            for luxury_kw in ["luxury", "premium", "resort", "5-star", "five-star"]:
                if luxury_kw in meta_str:
                    score += 0.5
                    break

        item_scores.append({
            "item_idx": item_idx,
            "name": str(name),
            "score": score,
            "metadata": metadata,
        })

    item_scores.sort(key=lambda x: x["score"], reverse=True)

    seen = set()
    recommendations = []
    for item in item_scores:
        identity = item["name"].strip().lower()
        if identity in seen:
            continue
        seen.add(identity)

        item_idx = item["item_idx"]
        item_embedding = rec.model.item_embedding(
            torch.tensor([item_idx], device=rec.device)
        )
        avg_score = item_embedding.norm().item()

        recommendations.append({
            "rank": len(recommendations) + 1,
            "destination": item["name"],
            "score": float(item["score"] + avg_score * 0.1),
            "content_score": float(item["score"]),
            "metadata": item["metadata"],
        })
        if len(recommendations) >= top_k:
            break

    return recommendations


@router.get("/health")
async def recommender_health():
    rec = _get_recommender()
    if rec is None:
        return {"status": "unavailable", "detail": "Modelo recomendador no cargado"}
    return {
        "status": "online",
        "num_users": len(rec.idx_to_user),
        "num_items": len(rec.idx_to_item),
    }


@router.post("/recommend")
async def recommend(request: PreferenceRequest):
    rec = _get_recommender()
    if rec is None:
        raise HTTPException(status_code=503, detail="Modelo recomendador no disponible")

    if request.user_id and request.user_id in rec.user_to_idx:
        try:
            results = rec.recommend(user_id=request.user_id, top_k=request.top_k)
            return {
                "mode": "personalized",
                "user_id": request.user_id,
                "recommendations": results,
            }
        except KeyError:
            pass

    preferences = {
        "trip_type": request.trip_type,
        "budget": request.budget,
        "interests": request.interests,
    }
    recommendations = _content_based_recommend(rec, preferences, request.top_k)

    return {
        "mode": "preference_based",
        "preferences": preferences,
        "recommendations": recommendations,
    }
