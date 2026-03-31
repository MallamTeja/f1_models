import os
import time
import json
import numpy as np
import xgboost as xgb
import joblib
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from contextlib import asynccontextmanager

ml_models = {}
lookup_data = {}

def load_model_artifact(file_path: str) -> Optional[Any]:
    """Helper to load model or artifact dictionary."""
    if not os.path.exists(file_path):
        return None
    try:
        artifact = joblib.load(file_path)
        if isinstance(artifact, dict) and "model" in artifact:
            return artifact
        return {"model": artifact, "imputer": None}
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def get_race_key_from_filename(filename: str) -> str:
    """Extract race key from filename (e.g., 'abu_dhabi_model.joblib' -> 'abudhabi')."""
    name = filename.lower().replace("_model.joblib", "").replace("_", "").replace("-", "")
    if name == "us":
        return "usa"
    return name

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Dynamically load all .joblib models from models/ directory
    base_path = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_path, "models")
    
    if os.path.exists(models_dir):
        for filename in os.listdir(models_dir):
            if filename.endswith(".joblib"):
                race_key = get_race_key_from_filename(filename)
                path = os.path.join(models_dir, filename)
                artifact = load_model_artifact(path)
                if artifact:
                    ml_models[race_key] = artifact
                    print(f"Loaded model for {race_key} from {filename}")
            
    # Load lookup data
    lookup_path = os.path.join(models_dir, "lookup_data.json")
    if os.path.exists(lookup_path):
        with open(lookup_path, "r") as f:
            lookup_data["data"] = json.load(f)
    
    yield
    ml_models.clear()

app = FastAPI(
    title="F1 Race Pace Predictor",
    description="API for predicting F1 race pace based on qualifying and weather data",
    version="1.3.0",
    lifespan=lifespan,
    docs_url="/",
    redoc_url=None
)

class PredictionInput(BaseModel):
    race_name: str = Field(description="Race name: 'abudhabi', 'qatar', 'usa', or 'mexico'")
    driver_code: str = Field(min_length=3, max_length=3, description="3-letter F1 driver code")
    qualifying_time: float = Field(gt=0, le=200, description="Qualifying lap time in seconds")
    clean_air_race_pace: float = Field(gt=0, le=200, description="Race pace with clean air in seconds")
    rain_prob: float = Field(ge=0, le=100, description="Rain probability as percentage")
    temperature: float = Field(ge=-10, le=70, description="Track temperature in Celsius")

    @field_validator("race_name")
    @classmethod
    def validate_race_name(cls, v: str) -> str:
        val = v.lower().strip().replace(" ", "_").replace("-", "_")
        if val in ["abudhabi", "abu_dhabi", "yas_marina"]:
            return "abudhabi"
        if val in ["qatar", "lusail"]:
            return "qatar"
        if val in ["usa", "united_states", "austin", "cota"]:
            return "usa"
        if val in ["mexico", "mexico_city"]:
            return "mexico"
        # Enhanced formal error message
        raise ValueError(
            f"The provided race name '{v}' is not valid. "
            "Please specify one of the supported race identifiers: "
            "'abudhabi', 'qatar', 'usa', or 'mexico'."
        )

@app.post("/predict")
async def predict(input_data: PredictionInput):
    start_time = time.time()
    race = input_data.race_name
    artifact = ml_models.get(race)
    
    if artifact is None:
        raise HTTPException(status_code=500, detail=f"Model for '{race}' not loaded")
    
    model = artifact["model"]
    imputer = artifact.get("imputer")
    
    drivers = lookup_data.get("data", {}).get("drivers", {})
    driver_code_upper = input_data.driver_code.upper()
    
    if driver_code_upper not in drivers:
        raise HTTPException(status_code=422, detail=f"Unknown driver code '{driver_code_upper}'")
    
    team_score = drivers[driver_code_upper]
    
    ranges = {
        "abudhabi": (70, 105),
        "qatar": (75, 120),
        "usa": (85, 130),
        "mexico": (70, 110)
    }
    valid_range = ranges.get(race)
    
    if not (valid_range[0] <= input_data.qualifying_time <= valid_range[1]):
        raise HTTPException(status_code=422, detail=f"Qualifying time for {race} invalid")
    
    if not (valid_range[0] <= input_data.clean_air_race_pace <= valid_range[1]):
        raise HTTPException(status_code=422, detail=f"Clean air race pace for {race} invalid")
    
    if input_data.clean_air_race_pace <= input_data.qualifying_time:
        raise HTTPException(status_code=422, detail="Clean air race pace should be slower than qualifying time")
    
    # Feature engineering based on model requirements
    # Note: USA and Mexico models use: QualifyingTime, CleanAirRacePace, TeamPerformanceScore, TotalSectorTime (imputed), RainProbability
    # Abu Dhabi and Qatar use: QualifyingTime, RainProbability, Temperature, TeamPerformanceScore, CleanAirRacePace
    
    if race in ["usa", "mexico"]:
        features = np.array([[
            input_data.qualifying_time,
            input_data.clean_air_race_pace,
            team_score,
            np.nan, # TotalSectorTime to be imputed
            input_data.rain_prob
        ]])
    else:
        features = np.array([[
            input_data.qualifying_time, 
            input_data.rain_prob, 
            input_data.temperature, 
            team_score, 
            input_data.clean_air_race_pace
        ]])
    
    try:
        if imputer:
            features = imputer.transform(features)
            
        if race in ["abudhabi", "qatar"] and hasattr(model, "predict") and "xgboost" in str(type(model)).lower():
            # Handle native XGBoost Booster if needed, though XGBRegressor is usually used
            if not hasattr(model, "predict"):
                 dmatrix = xgb.DMatrix(features)
                 prediction = model.predict(dmatrix)[0]
            else:
                 prediction = model.predict(features)[0]
            model_info = f"{race}_xgb_v2"
        else: 
            prediction = model.predict(features)[0]
            model_info = f"{race}_v2"

        latency = time.time() - start_time
        return {
            "race": race,
            "driver": driver_code_upper,
            "predicted_pace": float(prediction),
            "meta": {
                "latency": f"{latency:.4f}s",
                "model": model_info
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/info", include_in_schema=False)
async def info():
    return {
        "message": "F1 Race Pace Predictor API ",
        "version": "1.3.0",
        "available_races": ["Abu Dhabi", "Qatar", "United States", "Mexico"]
    }

@app.get("/health", include_in_schema=False)
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": list(ml_models.keys())
    }