import os
import time
import json
import numpy as np
import xgboost as xgb
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from contextlib import asynccontextmanager
from sklearn_json import from_dict
from sklearn_json import regression as reg
from sklearn.tree._tree import Tree
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import dummy

def patched_deserialize_tree(tree_dict, n_features, n_classes, n_outputs):
    tree_dict['nodes'] = [tuple(lst) for lst in tree_dict['nodes']]
    names = ['left_child', 'right_child', 'feature', 'threshold', 'impurity', 'n_node_samples', 'weighted_n_node_samples']
    if len(tree_dict['nodes'][0]) > 7:
        names.append('missing_go_to_left')
    tree_dict['nodes'] = np.array(tree_dict['nodes'], dtype=np.dtype({'names': names, 'formats': tree_dict['nodes_dtype']}))
    tree_dict['values'] = np.array(tree_dict['values'])
    tree = Tree(n_features, np.array([n_classes], dtype=np.intp), n_outputs)
    tree.__setstate__(tree_dict)
    return tree

def patched_deserialize_gradient_boosting_regressor(model_dict):
    model = GradientBoostingRegressor(**model_dict['params'])
    trees = [reg.deserialize_decision_tree_regressor(tree) for tree in model_dict['estimators_']]
    model.estimators_ = np.array(trees).reshape(model_dict['estimators_shape'])
    if 'init_' in model_dict and model_dict['init_']['meta'] == 'dummy':
        model.init_ = dummy.DummyRegressor()
        model.init_.__dict__ = model_dict['init_']
        model.init_.__dict__.pop('meta', None)

    model.train_score_ = np.array(model_dict['train_score_'])
    model.max_features_ = model_dict['max_features_']
    model.n_features_ = model_dict['n_features_']
    model.loss_ = None
    if 'priors' in model_dict:
        model.init_.priors = np.array(model_dict['priors'])
    return model

reg.deserialize_tree = patched_deserialize_tree
reg.deserialize_gradient_boosting_regressor = patched_deserialize_gradient_boosting_regressor

ml_models = {}
lookup_data = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.path.exists("abudhabi_model.json"):
        try:    
            booster = xgb.Booster()
            booster.load_model("abudhabi_model.json")
            ml_models["abudhabi"] = booster
        except Exception as e:
            pass
            
    if os.path.exists("qatarmodel.json"):
        try:
            booster = xgb.Booster()
            booster.load_model("qatarmodel.json")
            ml_models["qatar"] = booster
        except Exception as e:
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
        except Exception as e:
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
        except Exception as e:
            pass
    
    if os.path.exists("lookup_data.json"):
        with open("lookup_data.json", "r") as f:
            lookup_data["data"] = json.load(f)
    
    yield
    ml_models.clear()

app = FastAPI(
    title="F1 Race Pace Predictor",
    description="API for predicting F1 race pace based on qualifying and weather data",
    version="1.2.0",
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
        raise ValueError("Race name must be one of: abudhabi, qatar, usa, mexico")

@app.post("/predict")
async def predict(input_data: PredictionInput):
    start_time = time.time()
    race = input_data.race_name
    model = ml_models.get(race)
    
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model for '{race}' not loaded")
    
    drivers = lookup_data.get("data", {}).get("drivers", {})
    driver_code_upper = input_data.driver_code.upper()
    
    if driver_code_upper not in drivers:
        raise HTTPException(status_code=422, detail=f"Unknown driver code '{driver_code_upper}'")
    
    team_score = drivers[driver_code_upper]
    
    ranges = {
        "abudhabi": (70, 95),
        "qatar": (75, 110),
        "usa": (85, 120),
        "mexico": (70, 100)
    }
    valid_range = ranges.get(race)
    
    if not (valid_range[0] <= input_data.qualifying_time <= valid_range[1]):
        raise HTTPException(status_code=422, detail=f"Qualifying time for {race} invalid")
    
    if not (valid_range[0] <= input_data.clean_air_race_pace <= valid_range[1]):
        raise HTTPException(status_code=422, detail=f"Clean air race pace for {race} invalid")
    
    if input_data.clean_air_race_pace <= input_data.qualifying_time:
        raise HTTPException(status_code=422, detail="Clean air race pace should be slower than qualifying time")
    
    features = np.array([[
        input_data.qualifying_time, 
        input_data.rain_prob, 
        input_data.temperature, 
        team_score, 
        input_data.clean_air_race_pace
    ]])
    
    try:
        if race in ["abudhabi", "qatar"]:
            dmatrix = xgb.DMatrix(features)
            prediction = model.predict(dmatrix)[0]
            model_info = f"{race}_xgb_v1"
        else: 
            prediction = model.predict(features)[0]
            model_info = f"{race}_gbr_v1"

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
        "version": "1.2.0",
        "available_races": ["Abu Dhabi", "Qatar", "United States", "Mexico"]
    }

@app.get("/health", include_in_schema=False)
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": list(ml_models.keys())
    }