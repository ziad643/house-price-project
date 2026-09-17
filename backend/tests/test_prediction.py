from fastapi.testclient import TestClient

from app.main import app


class FakePredictor:
    locations = {"Andheri", "other"}

    def predict(self, frame):
        return 4_250_000.0


def test_prediction_happy_path():
    with TestClient(app) as client:
        app.state.predictor = FakePredictor()
        response = client.post("/predict", json={
            "location": "Andheri", "carpet_area_sqft": 850, "floor_num": 3,
            "bathroom": 2, "balcony": 1, "car_parking": 1, "furnishing": "Semi-Furnished",
            "transaction": "Resale", "ownership": "Freehold", "facing": "East",
        })
    assert response.status_code == 200
    assert response.json()["predicted_price"] == 4_250_000.0


def test_prediction_rejects_non_positive_area():
    with TestClient(app) as client:
        response = client.post("/predict", json={
            "location": "Andheri", "carpet_area_sqft": 0, "floor_num": 3,
            "bathroom": 2, "balcony": 1, "car_parking": 1, "furnishing": "Semi-Furnished",
            "transaction": "Resale", "ownership": "Freehold", "facing": "East",
        })
    assert response.status_code == 422
