import pytest
from fastapi.testclient import TestClient
from main import app, ml_models, lookup_data
import xgboost as xgb
import json
import os

if os.path.exists("abu_dhabi_model.json"):
    try:
        booster = xgb.Booster()
        booster.load_model("abu_dhabi_model.json")
        ml_models["f1_model"] = booster
    except:
        pass

if os.path.exists("lookup_data.json"):
    with open("lookup_data.json", "r") as f:
        lookup_data["data"] = json.load(f)

client = TestClient(app)

def test_read_root():
    response = client.get("/info")
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
    assert response.status_code == 200
    assert "VER" in response.json()["driver"]
    assert "predicted_pace" in response.json()

def test_predict_invalid_driver():
    payload = {
        "driver_code": "XXX",
        "qualifying_time": 82.207,
        "clean_air_race_pace": 91.10,
        "rain_prob": 0.0,
        "temperature": 25.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "Unknown driver code" in response.json()["detail"]


def test_predict_invalid_qualifying_time_zero():
    payload = {
        "driver_code": "VER",
        "qualifying_time": 0,
        "clean_air_race_pace": 91.10,
        "rain_prob": 0.0,
        "temperature": 25.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_rain_prob():
    payload = {
        "driver_code": "VER",
        "qualifying_time": 82.207,
        "clean_air_race_pace": 91.10,
        "rain_prob": 150.0,
        "temperature": 25.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_realistic_range_check():
    payload = {
        "driver_code": "VER",
        "qualifying_time": 150.0,
        "clean_air_race_pace": 160.0,
        "rain_prob": 0.0,
        "temperature": 25.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "70-95 seconds" in response.json()["detail"]


def test_predict_string_driver_code():
    payload = {
        "driver_code": "string",
        "qualifying_time": 82.207,
        "clean_air_race_pace": 91.10,
        "rain_prob": 0.0,
        "temperature": 25.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
