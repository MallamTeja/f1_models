# json parses lookup_data.json (driver -> team score table) at startup
import json

# logging replaces print so output carries levels and reaches uvicorn/Render log streams
import logging

# os.path keeps the artifact loader's existence check unchanged for callers passing str paths
import os

# time measures per-request latency reported back in the response meta
import time

# asynccontextmanager turns the generator below into FastAPI's startup/shutdown hook
from contextlib import asynccontextmanager

# Any types the loader's loosely shaped artifacts (dict or bare model)
from typing import Any

# joblib is the format every model in models/ was saved with
import joblib

# numpy builds the 2-D feature row the sklearn/xgboost models expect
import numpy as np

# FastAPI core: the app, HTTP errors for bad input, and the raw request for the 405 handler
from fastapi import FastAPI, HTTPException, Request

# JSONResponse lets handlers pick a status code (405 body, 503 health) without raising
from fastapi.responses import JSONResponse

# pydantic validates and normalises the request body before any model code runs
from pydantic import BaseModel, Field, field_validator

# config holds env-overridable paths and the single version string
from config import APP_VERSION, MODELS_DIR

# basicConfig is a no-op if a host already configured logging; otherwise makes INFO lines visible
logging.basicConfig(level=logging.INFO)
# Module-named logger so log lines show which file emitted them
logger = logging.getLogger(__name__)

# Race key -> {"model", "imputer"}; module-level so the lifespan fills it and handlers read it
ml_models = {}
# Holds parsed lookup_data.json under "data"; a dict so tests can clear/restore it in place
lookup_data = {}


def load_model_artifact(file_path: str) -> Any | None:
    """Helper to load model or artifact dictionary."""
    # Missing file is a normal case (partial deploy), so signal it with None instead of raising
    if not os.path.exists(file_path):
        return None
    # A corrupt or version-incompatible pickle must not crash startup for the other races
    try:
        # Unpickle once; the object may be a dict bundle or a bare estimator
        artifact = joblib.load(file_path)
        # usa/mexico were saved as {"model", "imputer"} bundles, so pass them through as-is
        if isinstance(artifact, dict) and "model" in artifact:
            return artifact
        # qatar/abudhabi are bare XGB models; wrap them so predict() sees one shape
        return {"model": artifact, "imputer": None}
    # Unpickling can raise almost anything (version skew, truncated file), so catching broadly is the point
    except Exception as e:  # noqa: BLE001
        # Log with the path so a bad pickle is traceable from deploy logs
        logger.error("Error loading %s: %s", file_path, e)
        # None lets the caller skip this race and keep loading the rest
        return None


def get_race_key_from_filename(filename: str) -> str:
    """Extract race key from filename (e.g., 'abu_dhabi_model.joblib' -> 'abudhabi')."""
    # Strip suffix and separators so the key matches what validate_race_name returns
    name = filename.lower().replace("_model.joblib", "").replace("_", "").replace("-", "")
    # The USA file is named us_model.joblib but the API key is "usa"
    if name == "us":
        return "usa"
    # Every other filename already yields its API key
    return name


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loading at startup (not per request) keeps /predict latency to inference only
    if MODELS_DIR.exists():
        # sorted() makes load order, and so log order, deterministic across filesystems
        for path in sorted(MODELS_DIR.glob("*.joblib")):
            # Derive the API key from the filename so new races need no code change here
            race_key = get_race_key_from_filename(path.name)
            # str() because the loader's contract is a plain path string
            artifact = load_model_artifact(str(path))
            # Only register races whose pickle actually loaded
            if artifact:
                # Make the model reachable by /predict
                ml_models[race_key] = artifact
                # Startup log line confirms which models a deploy actually has
                logger.info("Loaded model for %s from %s", race_key, path.name)

    # Driver -> team performance score table used as a model feature
    lookup_path = MODELS_DIR / "lookup_data.json"
    # Absent lookup file means every driver is "unknown" (422), not a crash
    if lookup_path.exists():
        # One-shot read before serving traffic, so blocking the loop briefly is harmless;
        # explicit encoding so the Windows cp1252 default cannot misread the file
        lookup_data["data"] = json.loads(lookup_path.read_text(encoding="utf-8"))

    # Hand control to FastAPI; everything after runs on shutdown
    yield
    # Release model memory on shutdown and leave a clean slate for the next TestClient start
    ml_models.clear()


# docs_url="/" puts Swagger on the root so the Render URL is self-describing
app = FastAPI(
    title="F1 Race Pace Predictor",
    description="API for predicting F1 race pace based on qualifying and weather data",
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/",
    redoc_url=None,
)


@app.exception_handler(405)
async def method_not_allowed_handler(request: Request, exc):
    # People hitting /predict from a browser send GET; tell them to POST instead of a bare 405
    return JSONResponse(
        status_code=405,
        content={
            "status": "error",
            "message": "Custom Error: This endpoint requires a POST request. Please send a JSON body using the POST method.",
            "path": request.url.path,
        },
    )


class PredictionInput(BaseModel):
    # Free-form race name; the validator below maps aliases to the 4 model keys
    race_name: str = Field(description="Race name: 'abudhabi', 'qatar', 'usa', or 'mexico'")
    # F1 codes are always 3 letters; enforce it before the lookup table check
    driver_code: str = Field(min_length=3, max_length=3, description="3-letter F1 driver code")
    # Coarse sanity bound; per-race ranges are enforced in predict()
    qualifying_time: float = Field(gt=0, le=200, description="Qualifying lap time in seconds")
    # Same coarse bound as qualifying; per-race ranges are enforced in predict()
    clean_air_race_pace: float = Field(gt=0, le=200, description="Race pace with clean air in seconds")
    # Percentage, so 0-100
    rain_prob: float = Field(ge=0, le=100, description="Rain probability as percentage")
    # Plausible track temperature range; rejects unit mistakes like Fahrenheit
    temperature: float = Field(ge=-10, le=70, description="Track temperature in Celsius")

    @field_validator("race_name")
    @classmethod
    def validate_race_name(cls, v: str) -> str:
        # Normalise case/spaces/dashes so "Abu Dhabi" and "abu-dhabi" hit the same alias
        val = v.lower().strip().replace(" ", "_").replace("-", "_")
        # Abu Dhabi aliases, including the circuit name
        if val in ["abudhabi", "abu_dhabi", "yas_marina"]:
            return "abudhabi"
        # Qatar aliases, including the circuit name
        if val in ["qatar", "lusail"]:
            return "qatar"
        # USA aliases, including city and circuit
        if val in ["usa", "united_states", "austin", "cota"]:
            return "usa"
        # Mexico aliases
        if val in ["mexico", "mexico_city"]:
            return "mexico"
        # Unknown race: pydantic turns this into a 422 with the message below
        raise ValueError(
            f"The provided race name '{v}' is not valid. "
            "Please specify one of the supported race identifiers: "
            "'abudhabi', 'qatar', 'usa', or 'mexico'."
        )


@app.post("/predict")
async def predict(input_data: PredictionInput):
    # Start the clock first so reported latency covers validation-to-response
    start_time = time.time()
    # Already normalised by the validator to one of the 4 keys
    race = input_data.race_name
    # None when that race's pickle failed to load or is missing
    artifact = ml_models.get(race)

    # Valid race but no model is a server-side fault, hence 500 not 422
    if artifact is None:
        raise HTTPException(status_code=500, detail=f"Model for '{race}' not loaded")

    # The estimator itself (sklearn pipeline or XGBRegressor)
    model = artifact["model"]
    # usa/mexico need TotalSectorTime imputed; bare XGB models carry None
    imputer = artifact.get("imputer")

    # Driver -> team score map; empty if lookup file was missing
    drivers = lookup_data.get("data", {}).get("drivers", {})
    # Lookup table keys are uppercase codes
    driver_code_upper = input_data.driver_code.upper()

    # Without a team score the feature row cannot be built, so reject as bad input
    if driver_code_upper not in drivers:
        raise HTTPException(status_code=422, detail=f"Unknown driver code '{driver_code_upper}'")

    # Team performance score feature for this driver
    team_score = drivers[driver_code_upper]

    # Per-circuit plausible lap time windows; outside them the model extrapolates badly
    ranges = {
        "abudhabi": (70, 105),
        "qatar": (75, 120),
        "usa": (85, 130),
        "mexico": (70, 110),
    }
    # Every validated race has an entry, so this is never None
    valid_range = ranges.get(race)

    # Reject qualifying laps the model never saw in training
    if not (valid_range[0] <= input_data.qualifying_time <= valid_range[1]):
        raise HTTPException(status_code=422, detail=f"Qualifying time for {race} invalid")

    # Reject race pace values the model never saw in training
    if not (valid_range[0] <= input_data.clean_air_race_pace <= valid_range[1]):
        raise HTTPException(status_code=422, detail=f"Clean air race pace for {race} invalid")

    # Race pace on full fuel is physically slower than a qualifying lap
    if input_data.clean_air_race_pace <= input_data.qualifying_time:
        raise HTTPException(status_code=422, detail="Clean air race pace should be slower than qualifying time")

    # Feature engineering based on model requirements
    # Note: USA and Mexico models use: QualifyingTime, CleanAirRacePace, TeamPerformanceScore, TotalSectorTime (imputed), RainProbability
    # Abu Dhabi and Qatar use: QualifyingTime, RainProbability, Temperature, TeamPerformanceScore, CleanAirRacePace

    # Column order must match each model's training order exactly
    if race in ["usa", "mexico"]:
        # usa/mexico feature order, with NaN for the imputer to fill
        features = np.array(
            [
                [
                    input_data.qualifying_time,
                    input_data.clean_air_race_pace,
                    team_score,
                    np.nan,  # TotalSectorTime to be imputed
                    input_data.rain_prob,
                ]
            ]
        )
    else:
        # abudhabi/qatar feature order; these models were trained with temperature
        features = np.array(
            [
                [
                    input_data.qualifying_time,
                    input_data.rain_prob,
                    input_data.temperature,
                    team_score,
                    input_data.clean_air_race_pace,
                ]
            ]
        )

    # Any model-side failure becomes a 500 with the reason, not an unhandled crash
    try:
        # Fill the NaN TotalSectorTime column for models that were trained with an imputer
        if imputer:
            features = imputer.transform(features)

        # All loaded models expose sklearn-style predict(); take the single row's value
        prediction = model.predict(features)[0]
        # Label kept identical to previous releases; clients and tests key off it
        if race in ["abudhabi", "qatar"] and "xgboost" in str(type(model)).lower():
            model_info = f"{race}_xgb_v2"
        else:
            model_info = f"{race}_v2"

        # Measure after inference so latency reflects real work
        latency = time.time() - start_time
        # float() because numpy scalars are not JSON serialisable
        return {
            "race": race,
            "driver": driver_code_upper,
            "predicted_pace": float(prediction),
            "meta": {"latency": f"{latency:.4f}s", "model": model_info},
        }
    # Model internals can raise any type; all of them must map to one 500 shape
    except Exception as e:
        # Surface the reason so a broken model is diagnosable from the client side
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}") from e


@app.get("/info", include_in_schema=False)
async def info():
    # Human-readable service summary; version comes from config so it matches OpenAPI
    return {
        "message": "F1 Race Pace Predictor API ",
        "version": APP_VERSION,
        "available_races": ["Abu Dhabi", "Qatar", "United States", "Mexico"],
    }


@app.get("/health", include_in_schema=False)
async def health_check():
    # No models means /predict can only 500, so report unhealthy for Docker HEALTHCHECK/Render
    if not ml_models:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "models_loaded": []})
    # Healthy body lists loaded races so a deploy check can confirm all 4 are present
    return {"status": "healthy", "models_loaded": list(ml_models.keys())}
