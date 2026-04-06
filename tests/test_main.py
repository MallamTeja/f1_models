import pytest
import os
import json
import joblib
import numpy as np
from fastapi.testclient import TestClient
from main import app, ml_models, lookup_data, load_model_artifact

# Standardized setup for tests
def setup_module(module):
    # Only try to load if not already loaded (lifespan usually handles this but for tests we might want isolation)
    model_configs = {
        "abudhabi": "models/abu_dhabi_model.joblib",
        "qatar": "models/qatar_model.joblib",
        "usa": "models/us_model.joblib",
        "mexico": "models/mexico_model.joblib"
    }
    
    for race, path in model_configs.items():
        if os.path.exists(path):
            ml_models[race] = load_model_artifact(path)
            
    lookup_path = "models/lookup_data.json"
    if os.path.exists(lookup_path):
        with open(lookup_path, "r") as f:
            lookup_data["data"] = json.load(f)

client = TestClient(app)

def test_read_root():
    response = client.get("/info")
    assert response.status_code == 200
    assert "F1 Race Pace Predictor" in response.json()["message"]
    assert response.json()["version"] == "1.3.0"

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_abudhabi():
    if "abudhabi" not in ml_models:
        pytest.skip("Abu Dhabi model not available")
    payload = {
        "race_name": "abudhabi",
        "driver_code": "VER",
        "qualifying_time": 82.207,
        "clean_air_race_pace": 91.10,
        "rain_prob": 0.0,
        "temperature": 25.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["race"] == "abudhabi"
    assert "xgb_v2" in data["meta"]["model"]

def test_predict_qatar():
    if "qatar" not in ml_models:
        pytest.skip("Qatar model not available")
    payload = {
        "race_name": "qatar",
        "driver_code": "VER",
        "qualifying_time": 82.207,
        "clean_air_race_pace": 93.20,
        "rain_prob": 0.0,
        "temperature": 30.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["race"] == "qatar"
    assert "xgb_v2" in data["meta"]["model"]

def test_predict_usa():
    if "usa" not in ml_models:
        pytest.skip("USA model not available")
    payload = {
        "race_name": "usa",
        "driver_code": "VER",
        "qualifying_time": 94.5,
        "clean_air_race_pace": 100.2,
        "rain_prob": 0.0,
        "temperature": 35.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["race"] == "usa"
    assert "v2" in data["meta"]["model"]



def test_predict_invalid_driver():
    payload = {
        "race_name": "abudhabi",
        "driver_code": "XXX",
        "qualifying_time": 82.207,
        "clean_air_race_pace": 91.10,
        "rain_prob": 0.0,
        "temperature": 25.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_invalid_qualifying_time():
    payload = {
        "race_name": "abudhabi",
        "driver_code": "VER",
        "qualifying_time": 20.0,
        "clean_air_race_pace": 91.10,
        "rain_prob": 0.0,
        "temperature": 25.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_invalid_race():
    payload = {
        "race_name": "moon",
        "driver_code": "VER",
        "qualifying_time": 82.207,
        "clean_air_race_pace": 91.10,
        "rain_prob": 0.0,
        "temperature": 25.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_logic_order():
    payload = {
        "race_name": "usa",
        "driver_code": "VER",
        "qualifying_time": 100.0,
        "clean_air_race_pace": 90.0,
        "rain_prob": 0.0,
        "temperature": 35.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "slower than qualifying time" in response.json()["detail"]
