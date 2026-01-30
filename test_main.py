import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "F1 Race Pace Predictor" in response.json()["message"]

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"




def test_predict_logic():
    payload = {
        "driver_code": "VER",
        "qualifying_time": 82.207,
        "clean_air_race_pace": 91.10,
        "rain_prob": 0.0,
        "temperature": 25.0
    }
    response = client.post("/predict", json=payload)
    print("STATUS:", response.status_code)
    print("BODY:", response.text)
    assert False
