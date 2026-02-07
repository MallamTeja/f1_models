import os
import time
import json
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
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
    lifespan=lifespan,
    docs_url="/",
    redoc_url=None
)

class PredictionInput(BaseModel):
    driver_code: str = Field(
        min_length=3,
        max_length=3,
        description="3-letter F1 driver code (e.g., VER, LEC, ALO)"
    )
    qualifying_time: float = Field(
        gt=0,
        le=200,
        description="Qualifying lap time in seconds"
    )
    clean_air_race_pace: float = Field(
        gt=0,
        le=200,
        description="Race pace with clean air in seconds"
    )
    rain_prob: float = Field(
        ge=0,
        le=100,
        description="Rain probability as percentage (0-100)"
    )
    temperature: float = Field(
        ge=-10,
        le=70,
        description="Track temperature in Celsius"
    )

@app.post("/predict")
async def predict(input_data: PredictionInput):
    start_time = time.time()
    model = ml_models.get("f1_model")
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    drivers = lookup_data.get("data", {}).get("drivers", {})
    driver_code_upper = input_data.driver_code.upper()
    
    if driver_code_upper not in drivers:
        allowed_drivers = ", ".join(sorted(drivers.keys()))
        raise HTTPException(
            status_code=422,
            detail=f"Unknown driver code '{driver_code_upper}'. Allowed: {allowed_drivers}"
        )
    
    team_score = drivers[driver_code_upper]
    
    if not (70 <= input_data.qualifying_time <= 95):
        raise HTTPException(
            status_code=422,
            detail="Qualifying time must be 70-95 seconds (realistic for Abu Dhabi)"
        )
    
    if not (70 <= input_data.clean_air_race_pace <= 95):
        raise HTTPException(
            status_code=422,
            detail="Clean air race pace must be 70-95 seconds"
        )
    
    if input_data.clean_air_race_pace <= input_data.qualifying_time:
        raise HTTPException(
            status_code=422,
            detail="Clean air race pace should be slower than qualifying time"
        )
    
    features = np.array([[input_data.qualifying_time,input_data.rain_prob,input_data.temperature,team_score,input_data.clean_air_race_pace]])
    
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

@app.get("/info", include_in_schema=False)
async def info():
    return {
        "message": "F1 Race Pace Predictor API",
        "version": "1.0.0"
    }

@app.get("/health", include_in_schema=False)
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": "f1_model" in ml_models
    }
