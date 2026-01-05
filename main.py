import os
import time
import json
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

ml_models = {}
lookup_data = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.path.exists("abu_dhabi_ensemble_model.joblib"):
        ml_models["f1_model"] = joblib.load("abu_dhabi_ensemble_model.joblib")
        print("Model loaded.")
    if os.path.exists("lookup_data.json"):
        with open("lookup_data.json", "r") as f:
            lookup_data["data"] = json.load(f)
            print("Lookup data loaded.")
    yield
    ml_models.clear()

app = FastAPI(
    title="F1 Race Pace Predictor",
    description="API for predicting F1 race pace based on qualifying times and weather conditions",
    version="1.0.0",
    lifespan=lifespan
)

class PredictionInput(BaseModel):
    driver_code: str
    qualifying_time: float
    clean_air_race_pace: float
    rain_prob: float = 0.0
    temperature: float = 25.0

@app.post("/predict")
async def predict(input_data: PredictionInput):
    start_time = time.time()
    model = ml_models.get("f1_model")
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    drivers = lookup_data.get("data", {}).get("drivers", {})
    team_score = drivers.get(input_data.driver_code.upper(), 0.5)
    features = np.array([[input_data.qualifying_time, input_data.rain_prob, input_data.temperature, team_score, input_data.clean_air_race_pace]])
    try:
        prediction = model.predict(features)[0]
        duration = time.time() - start_time
        return {
            "driver": input_data.driver_code,
            "predicted_pace": float(prediction),
            "meta": {
                "latency": f"{duration:.4f}s",
                "model": "f1_ensemble_v1.0"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "F1 Race Pace Predictor API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": "f1_model" in ml_models}
