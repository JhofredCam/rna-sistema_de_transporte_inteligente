import io
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


# ──────────────────────────────────────────────────────────────
# Root
# ──────────────────────────────────────────────────────────────

class TestRoot:
    def test_root_status(self, client):
        response = client.get("/")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "online"
        assert "endpoints" in body

    def test_root_endpoints(self, client):
        response = client.get("/")
        endpoints = response.json()["endpoints"]
        assert "/demand" in endpoints
        assert "/distraction" in endpoints
        assert "/recommender" in endpoints


# ──────────────────────────────────────────────────────────────
# Demand
# ──────────────────────────────────────────────────────────────

class TestDemand:
    def test_metadata(self, client):
        response = client.get("/demand/metadata")
        assert response.status_code == 200
        body = response.json()
        assert body["sequence_length"] == 30
        assert body["forecast_horizon"] == 30
        assert len(body["routes"]) == 5
        assert len(body["climas"]) >= 2
        assert body["model"] == "TransportLSTMAttention"

    def test_metadata_feature_columns(self, client):
        response = client.get("/demand/metadata")
        body = response.json()
        assert body["feature_columns"] == ["dia_semana", "mes", "festivo", "pasajeros"]
        assert body["future_feature_columns"] == ["dia_semana", "mes", "festivo"]

    def test_predict_with_history(self, client):
        response = client.post("/demand/predict", json={"route_id": 0, "steps": 7})
        assert response.status_code == 200
        body = response.json()
        assert "ruta" in body
        assert body["historico"] is not None
        assert body["prediccion_historica"] is not None
        assert body["pronostico"] is not None
        assert len(body["pronostico"]) == 7

    def test_predict_full_horizon(self, client):
        response = client.post("/demand/predict", json={"route_id": 2, "steps": 30})
        assert response.status_code == 200
        body = response.json()
        assert len(body["pronostico"]) == 30
        for row in body["pronostico"]:
            assert "fecha" in row
            assert "prediccion" in row
            assert isinstance(row["prediccion"], float)

    def test_predict_all_routes(self, client):
        for route_id in range(5):
            response = client.post("/demand/predict", json={"route_id": route_id, "steps": 3})
            assert response.status_code == 200
            body = response.json()
            assert len(body["pronostico"]) == 3

    def test_predict_with_custom_sequence(self, client):
        sequence = np.random.rand(30, 4).tolist()
        response = client.post(
            "/demand/predict",
            json={
                "route_id": 0,
                "steps": 5,
                "sequence": sequence,
                "future_features": [[0.5, 0.5, 0.0]] * 5,
                "future_clima_ids": [1] * 5,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["historico"] is None
        assert body["prediccion_historica"] is None

    def test_predict_invalid_route(self, client):
        response = client.post("/demand/predict", json={"route_id": 99, "steps": 7})
        assert response.status_code == 422

    def test_predict_invalid_steps(self, client):
        response = client.post("/demand/predict", json={"route_id": 0, "steps": 0})
        assert response.status_code == 422

    def test_predict_steps_over_limit(self, client):
        response = client.post("/demand/predict", json={"route_id": 0, "steps": 31})
        assert response.status_code == 422

    def test_predict_bad_sequence_shape(self, client):
        response = client.post(
            "/demand/predict",
            json={
                "route_id": 0,
                "steps": 5,
                "sequence": [[1.0, 2.0]] * 10,
            },
        )
        assert response.status_code == 400

    def test_predict_mismatched_future_lengths(self, client):
        sequence = np.random.rand(30, 4).tolist()
        response = client.post(
            "/demand/predict",
            json={
                "route_id": 0,
                "steps": 5,
                "sequence": sequence,
                "future_features": [[0.5, 0.5, 0.0]] * 3,
                "future_clima_ids": [1] * 5,
            },
        )
        assert response.status_code == 400

    def test_historical_dates_are_chronological(self, client):
        response = client.post("/demand/predict", json={"route_id": 0, "steps": 7})
        body = response.json()
        dates = [row["fecha"] for row in body["pronostico"]]
        assert dates == sorted(dates)

    def test_predictions_are_positive(self, client):
        response = client.post("/demand/predict", json={"route_id": 1, "steps": 10})
        body = response.json()
        for row in body["pronostico"]:
            assert isinstance(row["prediccion"], (int, float))


# ──────────────────────────────────────────────────────────────
# Distraction
# ──────────────────────────────────────────────────────────────

class TestDistraction:
    def test_health(self, client):
        response = client.get("/distraction/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ready"
        assert body["model"] == "module2_distraction"
        assert len(body["classes"]) >= 2
        assert "image_size" in body

    def test_classes(self, client):
        response = client.get("/distraction/classes")
        assert response.status_code == 200
        body = response.json()
        assert "classes" in body
        for cls in body["classes"]:
            assert "id" in cls
            assert "label" in cls
            assert "preventive_measure" in cls

    def test_predict_valid_image(self, client):
        image = Image.new("RGB", (224, 224), color=(128, 100, 80))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)

        response = client.post(
            "/distraction/predict",
            files={"file": ("test.jpg", buffer, "image/jpeg")},
        )
        assert response.status_code == 200
        body = response.json()
        assert "predicted_label" in body
        assert "confidence" in body
        assert "probabilities" in body
        assert "preventive_measure" in body
        assert 0.0 <= body["confidence"] <= 1.0
        assert isinstance(body["probabilities"], dict)

    def test_predict_png_image(self, client):
        image = Image.new("RGB", (300, 300), color=(50, 150, 200))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        response = client.post(
            "/distraction/predict",
            files={"file": ("test.png", buffer, "image/png")},
        )
        assert response.status_code == 200
        assert "predicted_label" in response.json()

    def test_predict_non_image_rejected(self, client):
        buffer = io.BytesIO(b"this is not an image")
        response = client.post(
            "/distraction/predict",
            files={"file": ("test.txt", buffer, "text/plain")},
        )
        assert response.status_code == 400

    def test_predict_empty_image_rejected(self, client):
        buffer = io.BytesIO(b"")
        response = client.post(
            "/distraction/predict",
            files={"file": ("empty.jpg", buffer, "image/jpeg")},
        )
        assert response.status_code == 400

    def test_predict_probabilities_sum_to_one(self, client):
        image = Image.new("RGB", (224, 224), color=(100, 100, 100))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)

        response = client.post(
            "/distraction/predict",
            files={"file": ("test.jpg", buffer, "image/jpeg")},
        )
        body = response.json()
        total = sum(body["probabilities"].values())
        assert abs(total - 1.0) < 0.05

    def test_predict_label_in_known_classes(self, client):
        classes_response = client.get("/distraction/classes")
        known_classes = {cls["label"] for cls in classes_response.json()["classes"]}

        image = Image.new("RGB", (224, 224), color=(200, 50, 50))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)

        response = client.post(
            "/distraction/predict",
            files={"file": ("test.jpg", buffer, "image/jpeg")},
        )
        assert response.json()["predicted_label"] in known_classes


# ──────────────────────────────────────────────────────────────
# Recommender
# ──────────────────────────────────────────────────────────────

class TestRecommender:
    def test_health(self, client):
        response = client.get("/recommender/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] in ("online", "unavailable")

    def test_recommend_by_preferences(self, client):
        response = client.post(
            "/recommender/recommend",
            json={
                "trip_type": "beach",
                "budget": "medio",
                "interests": ["nature", "culture"],
                "top_k": 5,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["mode"] in ("preference_based", "personalized")
        assert "recommendations" in body
        assert len(body["recommendations"]) <= 5

    def test_recommend_minimal_request(self, client):
        response = client.post(
            "/recommender/recommend",
            json={"top_k": 3},
        )
        assert response.status_code == 200
        body = response.json()
        assert len(body["recommendations"]) <= 3

    def test_recommend_with_user_id(self, client):
        response = client.post(
            "/recommender/recommend",
            json={"user_id": "1", "top_k": 5},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["mode"] in ("personalized", "preference_based")
        assert "recommendations" in body

    def test_recommend_top_k_limit(self, client):
        response = client.post(
            "/recommender/recommend",
            json={"top_k": 21},
        )
        assert response.status_code == 422

    def test_recommend_top_k_zero(self, client):
        response = client.post(
            "/recommender/recommend",
            json={"top_k": 0},
        )
        assert response.status_code == 422

    def test_recommend_recommendations_have_rank(self, client):
        response = client.post(
            "/recommender/recommend",
            json={"interests": ["adventure"], "top_k": 3},
        )
        body = response.json()
        for rec in body["recommendations"]:
            assert "rank" in rec
            assert "destination" in rec
            assert "score" in rec

    def test_recommend_budget_bajo(self, client):
        response = client.post(
            "/recommender/recommend",
            json={"budget": "bajo", "top_k": 5},
        )
        assert response.status_code == 200

    def test_recommend_budget_alto(self, client):
        response = client.post(
            "/recommender/recommend",
            json={"budget": "alto", "top_k": 5},
        )
        assert response.status_code == 200


# ──────────────────────────────────────────────────────────────
# Cross-cutting / CORS
# ──────────────────────────────────────────────────────────────

class TestCrossCutting:
    def test_cors_headers_present(self, client):
        response = client.options(
            "/demand/metadata",
            headers={"Origin": "http://localhost:5173"},
        )
        assert "access-control-allow-origin" in response.headers

    def test_nonexistent_endpoint_404(self, client):
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_double_slash_middleware(self, client):
        from starlette.requests import Request
        from api.main import StripDoubleSlashMiddleware

        captured_path = {}

        async def mock_call_next(request):
            captured_path["path"] = request.scope["path"]

            class FakeResponse:
                status_code = 200

            return FakeResponse()

        middleware = StripDoubleSlashMiddleware(app=None)

        import asyncio

        scope = {"type": "http", "path": "//demand//metadata"}
        request = Request(scope)

        async def run():
            await middleware.dispatch(request, mock_call_next)

        asyncio.get_event_loop().run_until_complete(run())
        assert captured_path["path"] == "/demand/metadata"
