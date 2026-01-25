import os
import time
import json
import numpy as np
import xgboost as xgb

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager


ml_models = {}
lookup_data = {}


@asynccontextmanager
async def lifespan(app: FastAPI):

    if os.path.exists("abu_dhabi_model.json"):
        try:
            booster = xgb.Booster()
            booster.load_model("abu_dhabi_model.json")
            ml_models["f1_model"] = booster
            print("Model loaded.")
        except Exception as e:
            print(f"Error loading model: {e}")
          
            try:
                with open("abu_dhabi_model.json", "rb") as f:
                    booster = xgb.Booster()
                    booster.load_model(f)
                    ml_models["f1_model"] = booster
                    print("Model loaded as binary.")
            except Exception as e2:
                print(f"Error loading model as binary: {e2}")


    if os.path.exists("lookup_data.json"):
        with open("lookup_data.json", "r") as f:
            lookup_data["data"] = json.load(f)
            print("Lookup data loaded.")

    yield

    ml_models.clear()


app = FastAPI(
    title="F1 Race Pace Predictor",
    description="API for predicting F1 race pace based on qualifying and weather data",
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

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    drivers = lookup_data.get("data", {}).get("drivers", {})
    team_score = drivers.get(input_data.driver_code.upper(), 0.5)

    features = np.array([[
        input_data.qualifying_time,
        input_data.rain_prob,
        input_data.temperature,
        team_score,
        input_data.clean_air_race_pace
    ]])

    try:
        dmatrix = xgb.DMatrix(features)
        prediction = model.predict(dmatrix)[0]

        latency = time.time() - start_time

        return {
            "driver": input_data.driver_code.upper(),
            "predicted_pace": float(prediction),
            "meta": {
                "latency": f"{latency:.4f}s",
                "model": "abu_dhabi_xgb_v1"
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {
        "message": "F1 Race Pace Predictor API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": "f1_model" in ml_models}
