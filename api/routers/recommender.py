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


_KEYWORD_MAP: dict[str, list[str]] = {
    "playa": ["beach", "coast", "sea", "water"],
    "montaña": ["mountain", "hill", "nature"],
    "ciudad": ["city", "urban"],
    "aventura": ["adventure"],
    "espiritual": ["heritage", "spiritual", "temple"],
    "gastronomía": ["food", "cuisine", "gastronomy"],
    "historia": ["historical", "history", "heritage"],
    "ecoturismo": ["nature", "eco", "wildlife"],
    "buceo": ["beach", "water", "sea", "diving"],
    "senderismo": ["adventure", "mountain", "trekking", "hiking"],
    "cultura": ["heritage", "historical", "culture", "city"],
    "naturaleza": ["nature"],
    "nature": ["nature"],
    "beach": ["beach", "coast", "sea", "water"],
    "adventure": ["adventure"],
    "culture": ["culture", "heritage", "historical", "city"],
    "city": ["city", "urban"],
    "mountain": ["mountain", "hill", "nature"],
    "historical": ["historical", "history", "heritage"],
    "heritage": ["heritage", "historical"],
}


def _expand_keywords(keywords: set[str]) -> set[str]:
    expanded: set[str] = set()
    for kw in keywords:
        kw_lower = kw.lower().strip()
        if kw_lower in _KEYWORD_MAP:
            expanded.update(_KEYWORD_MAP[kw_lower])
        else:
            expanded.add(kw_lower)
    return expanded


def _content_based_recommend(
    rec,
    preferences: dict[str, Any],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    raw_keywords: set[str] = set()
    if preferences.get("trip_type"):
        raw_keywords.add(preferences["trip_type"])
    if preferences.get("interests"):
        raw_keywords.update(preferences["interests"])

    expanded_keywords = _expand_keywords(raw_keywords)
    total_keywords = len(expanded_keywords) if expanded_keywords else 0

    items = []
    popularities = []
    for item_idx in range(len(rec.idx_to_item)):
        metadata = rec.item_metadata[item_idx] if item_idx < len(rec.item_metadata) else {}
        name = rec._display_name(item_idx)
        name_lower = str(name).lower()
        meta_str = " ".join(str(v) for v in metadata.values()).lower()
        type_lower = str(metadata.get("Type", "")).lower()

        match_count = 0
        for kw in expanded_keywords:
            if kw in meta_str or kw in name_lower or kw in type_lower:
                match_count += 1

        pop_raw = metadata.get("Popularity", 0)
        try:
            pop_val = float(pop_raw)
        except (TypeError, ValueError):
            pop_val = 0.0
        popularities.append(pop_val)

        items.append({
            "item_idx": item_idx,
            "name": str(name),
            "match_count": match_count,
            "popularity": pop_val,
            "metadata": metadata,
        })

    pop_min = min(popularities) if popularities else 0.0
    pop_max = max(popularities) if popularities else 1.0
    pop_range = (pop_max - pop_min) if pop_max > pop_min else 1.0

    budget = preferences.get("budget")
    for item in items:
        pop_norm = (item["popularity"] - pop_min) / pop_range

        if total_keywords == 0:
            item["score"] = pop_norm
            item["content_score"] = 0.0
        else:
            match_ratio = item["match_count"] / total_keywords
            item["content_score"] = match_ratio
            if match_ratio > 0:
                item["score"] = 0.6 + match_ratio * 0.4
            else:
                item["score"] = pop_norm * 0.4

        if budget == "bajo":
            item["score"] += (1.0 - pop_norm) * 0.1
        elif budget == "alto":
            item["score"] += pop_norm * 0.1

        item["score"] = max(0.0, min(1.0, item["score"]))

    items.sort(key=lambda x: (x["score"], x["popularity"]), reverse=True)

    seen: set[str] = set()
    recommendations: list[dict[str, Any]] = []
    for item in items:
        identity = item["name"].strip().lower()
        if identity in seen:
            continue
        seen.add(identity)

        recommendations.append({
            "rank": len(recommendations) + 1,
            "destination": item["name"],
            "score": round(float(item["score"]), 4),
            "content_score": round(float(item["content_score"]), 4),
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
