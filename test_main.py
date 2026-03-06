import pytest
from fastapi.testclient import TestClient
from main import app, ml_models, lookup_data
import xgboost as xgb
import json
import os
from sklearn_json import from_dict
import numpy as np

if os.path.exists("abudhabi_model.json"):
    try:
        booster = xgb.Booster()
        booster.load_model("abudhabi_model.json")
        ml_models["abudhabi"] = booster
    except:
        pass

if os.path.exists("qatarmodel.json"):
    try:
        booster = xgb.Booster()
        booster.load_model("qatarmodel.json")
        ml_models["qatar"] = booster
    except:
        pass

if os.path.exists("us_model.json"):
    try:
        with open("us_model.json", "r") as f:
            artifact = json.load(f)
            ml_models["usa"] = from_dict(artifact["model"])
            if not hasattr(ml_models["usa"], "_loss"):
                from sklearn.ensemble import GradientBoostingRegressor
                dummy = GradientBoostingRegressor().fit(np.zeros((1, 5)), np.zeros(1))
                ml_models["usa"]._loss = dummy._loss
    except:
        pass

if os.path.exists("mexico_model.json"):
    try:
        with open("mexico_model.json", "r") as f:
            artifact = json.load(f)
            ml_models["mexico"] = from_dict(artifact["model"])
            if not hasattr(ml_models["mexico"], "_loss"):
                from sklearn.ensemble import GradientBoostingRegressor
                dummy = GradientBoostingRegressor().fit(np.zeros((1, 5)), np.zeros(1))
                ml_models["mexico"]._loss = dummy._loss
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
    assert response.json()["version"] == "1.2.0"

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_abudhabi():
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
    assert data["meta"]["model"] == "abudhabi_xgb_v1"

def test_predict_qatar():
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
    assert data["meta"]["model"] == "qatar_xgb_v1"

def test_predict_usa():
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
    assert data["meta"]["model"] == "usa_gbr_v1"

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

def test_predict_mexico():
    payload = {
        "race_name": "mexico",
        "driver_code": "VER",
        "qualifying_time": 82.207,
        "clean_air_race_pace": 91.10,
        "rain_prob": 0.0,
        "temperature": 25.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["race"] == "mexico"
    assert data["meta"]["model"] == "mexico_gbr_v1"

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
