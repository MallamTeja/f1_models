# pytest supplies fixtures, parametrize and approx for float comparisons
import pytest

# TestClient drives the ASGI app in-process, so no server or port is needed
from fastapi.testclient import TestClient

# pydantic's error type is what the race-name validator raises on bad input
from pydantic import ValidationError

# Version is asserted against config, the single source, not a copied literal
from config import APP_VERSION

# Import the real app and its shared state so tests exercise what production runs
from main import PredictionInput, app, get_race_key_from_filename, ml_models

# Regression tolerance: catches pickle/library drift while ignoring float noise
TOLERANCE = 0.01


@pytest.fixture(scope="module")
def client():
    # Context manager runs the real lifespan, so tests load models exactly like uvicorn does
    with TestClient(app) as test_client:
        # Hand the started client to every test in this module; one model load for all
        yield test_client


def payload(race_name, qualifying_time, clean_air_race_pace, temperature, driver_code="VER", rain_prob=0.0):
    # One builder keeps each test focused on the fields it varies
    return {
        "race_name": race_name,
        "driver_code": driver_code,
        "qualifying_time": qualifying_time,
        "clean_air_race_pace": clean_air_race_pace,
        "rain_prob": rain_prob,
        "temperature": temperature,
    }


def test_info(client):
    # /info is the human-facing summary; it must answer and report the configured version
    response = client.get("/info")
    # Endpoint must be reachable without a body
    assert response.status_code == 200
    # Message identifies the service to whoever curls it
    assert "F1 Race Pace Predictor" in response.json()["message"]
    # Guards against the version string drifting from config
    assert response.json()["version"] == APP_VERSION


def test_openapi_version_matches_config(client):
    # FastAPI(version=...) and /info used to duplicate the literal; both must read config now
    assert client.get("/openapi.json").json()["info"]["version"] == APP_VERSION


def test_health_check(client):
    # With models loaded, health must be 200 so Docker/Render mark the container healthy
    response = client.get("/health")
    # Healthy status code is what the HEALTHCHECK actually checks
    assert response.status_code == 200
    # Body status mirrors the code for humans reading it
    assert response.json()["status"] == "healthy"
    # All four shipped pickles must load; a missing one means a broken deploy
    assert sorted(response.json()["models_loaded"]) == ["abudhabi", "mexico", "qatar", "usa"]


def test_health_unhealthy_when_no_models(client):
    # Snapshot loaded models so the rest of the module still has them afterwards
    saved = dict(ml_models)
    # Simulate a deploy where every pickle failed to load
    ml_models.clear()
    # try/finally restores state even if an assert fails
    try:
        # Probe health in the broken state
        response = client.get("/health")
        # 503 is what makes Docker HEALTHCHECK fail instead of reporting a useless container healthy
        assert response.status_code == 503
        # Body must say unhealthy so log readers see the same verdict
        assert response.json() == {"status": "unhealthy", "models_loaded": []}
    finally:
        # Put the models back for later tests
        ml_models.update(saved)


@pytest.mark.parametrize(
    # race, request inputs, expected pace measured through the API, expected model label
    ("race", "qual", "pace", "temp", "expected", "label"),
    [
        # Mexico: imputer + GBR bundle
        ("mexico", 94.5, 100.2, 25.0, 86.102, "mexico_v2"),
        # USA: pickled with sklearn 1.8.0, loaded under 1.5.2; this guards against output drift
        ("usa", 94.5, 100.2, 35.0, 102.095, "usa_v2"),
        # Qatar: bare XGB model, no imputer
        ("qatar", 82.2, 91.1, 25.0, 92.429, "qatar_xgb_v2"),
        # Abu Dhabi: bare XGB model, no imputer
        ("abudhabi", 82.2, 91.1, 25.0, 89.512, "abudhabi_xgb_v2"),
    ],
)
def test_predict_regression(client, race, qual, pace, temp, expected, label):
    # Same inputs must keep producing the same pace; a dependency bump that shifts it fails here
    response = client.post("/predict", json=payload(race, qual, pace, temp))
    # Valid input for a loaded model must succeed
    assert response.status_code == 200
    # Parse once for the field checks below
    data = response.json()
    # Response echoes the normalised race key
    assert data["race"] == race
    # Driver code is echoed uppercase
    assert data["driver"] == "VER"
    # Core regression check against the measured value
    assert data["predicted_pace"] == pytest.approx(expected, abs=TOLERANCE)
    # Model labels are part of the public response; clients may key off them
    assert data["meta"]["model"] == label


@pytest.mark.parametrize(
    # raw user input and the model key it must normalise to
    ("raw", "expected"),
    [
        # Display name with a space
        ("Abu Dhabi", "abudhabi"),
        # Snake-case alias
        ("abu_dhabi", "abudhabi"),
        # Circuit name alias
        ("yas_marina", "abudhabi"),
        # Circuit name alias for Qatar
        ("Lusail", "qatar"),
        # Circuit abbreviation alias for USA
        ("cota", "usa"),
        # Hyphenated full name
        ("united-states", "usa"),
        # City-name alias
        ("mexico_city", "mexico"),
        # Surrounding whitespace and caps must not matter
        ("  MEXICO ", "mexico"),
    ],
)
def test_race_name_aliases(raw, expected):
    # Validator is pure, so test it directly without HTTP overhead
    model = PredictionInput(**payload(raw, 82.2, 91.1, 25.0))
    # Alias must collapse to the key ml_models is indexed by
    assert model.race_name == expected


def test_race_alias_through_api(client):
    # End-to-end proof that a display name reaches the right model, not just the validator
    response = client.post("/predict", json=payload("Abu Dhabi", 82.2, 91.1, 25.0))
    # Alias must be accepted
    assert response.status_code == 200
    # And routed to the Abu Dhabi model
    assert response.json()["race"] == "abudhabi"


def test_unknown_race_name_rejected_by_validator():
    # Unknown names must fail validation rather than reach ml_models.get()
    with pytest.raises(ValidationError, match="not valid"):
        PredictionInput(**payload("moon", 82.2, 91.1, 25.0))


@pytest.mark.parametrize(
    # filename on disk and the API key the lifespan must register it under
    ("filename", "expected"),
    [
        # File is "us" but the API key is "usa"
        ("us_model.joblib", "usa"),
        # Underscore in the race name must be stripped
        ("abu_dhabi_model.joblib", "abudhabi"),
        # Plain name passes through
        ("qatar_model.joblib", "qatar"),
        # Plain name passes through
        ("mexico_model.joblib", "mexico"),
    ],
)
def test_get_race_key_from_filename(filename, expected):
    # A wrong key silently hides a model from /predict, so pin the mapping for every shipped file
    assert get_race_key_from_filename(filename) == expected


def test_predict_invalid_driver(client):
    # Driver absent from lookup_data has no team score, so it must be rejected as bad input
    response = client.post("/predict", json=payload("abudhabi", 82.207, 91.10, 25.0, driver_code="XXX"))
    # 422 = client error, not a server crash
    assert response.status_code == 422
    # Message names the bad code so the caller can fix it
    assert "Unknown driver code 'XXX'" in response.json()["detail"]


def test_predict_invalid_qualifying_time(client):
    # 20 s is outside Abu Dhabi's window; the model would extrapolate nonsense
    response = client.post("/predict", json=payload("abudhabi", 20.0, 91.10, 25.0))
    # Must be rejected before inference
    assert response.status_code == 422
    # Message points at the field that failed
    assert response.json()["detail"] == "Qualifying time for abudhabi invalid"


def test_predict_invalid_race(client):
    # Unknown race via HTTP must surface as 422 from pydantic
    response = client.post("/predict", json=payload("moon", 82.207, 91.10, 25.0))
    # Validation failure, not a 500 from a missing model
    assert response.status_code == 422


def test_predict_logic_order(client):
    # Race pace faster than qualifying is physically impossible and must be refused
    response = client.post("/predict", json=payload("usa", 100.0, 90.0, 35.0))
    # Rejected as bad input
    assert response.status_code == 422
    # Message explains the rule that was broken
    assert "slower than qualifying time" in response.json()["detail"]


def test_predict_get_returns_custom_405(client):
    # Browsers send GET to /predict; the custom handler must explain POST is required
    response = client.get("/predict")
    # Method not allowed
    assert response.status_code == 405
    # Custom body, not FastAPI's bare default
    assert "requires a POST request" in response.json()["message"]
