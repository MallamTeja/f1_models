# warnings records what /predict emits, so a test can prove the feature-name warning is gone
import warnings

# pytest supplies fixtures, parametrize and approx for float comparisons
import pytest

# TestClient drives the ASGI app in-process, so no server or port is needed
from fastapi.testclient import TestClient

# pydantic's error type is what the race-name validator raises on bad input
from pydantic import ValidationError

# Version is asserted against config, the single source, not a copied literal
from config import APP_VERSION

# Import the real app and its shared state so tests exercise what production runs
from main import (
    EXPECTED_FEATURES,
    PredictionInput,
    app,
    feature_names_match,
    get_race_key_from_filename,
    lookup_data,
    ml_models,
)

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
        # Qatar: imputer + XGB bundle
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


def test_post_health_405_names_get(client):
    # /health is GET-only; the 405 message must come from the route's Allow header, not a hardcoded "POST"
    response = client.post("/health")
    # Method not allowed
    assert response.status_code == 405
    # Parse once for the checks below
    message = response.json()["message"]
    # Must point the caller at the method that actually works here
    assert "GET" in message
    # And must not repeat the /predict advice, which is wrong for this route
    assert "POST" not in message


@pytest.mark.parametrize(
    # Raw JSON tokens Python's json module accepts but strict JSON cannot carry
    "bad_float",
    ["NaN", "Infinity", "-Infinity"],
)
def test_non_finite_float_returns_422_not_500(client, bad_float):
    # Built as raw text because json= would refuse to encode NaN; this is what a hostile client sends
    body = (
        '{"race_name": "qatar", "driver_code": "VER", '
        f'"qualifying_time": {bad_float}, "clean_air_race_pace": 91.1, '
        '"rain_prob": 0.0, "temperature": 25.0}'
    )
    # Content-type must be JSON so FastAPI parses the body instead of rejecting the media type
    response = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    # Default handler echoed the NaN input and crashed into a 500; this must be a clean 422
    assert response.status_code == 422
    # Parse once for the shape checks below
    errors = response.json()["detail"]
    # The failing field must still be reported so the caller can fix it
    assert any(e["loc"][-1] == "qualifying_time" for e in errors)
    # "input" is the key that carried the non-finite float; it must be gone
    assert all("input" not in e for e in errors)


def test_race_name_over_max_length_rejected(client):
    # 33 chars is one past the cap; oversized input must stop before the validator echoes it
    response = client.post("/predict", json=payload("a" * 33, 82.2, 91.1, 25.0))
    # Validation failure, not a server error
    assert response.status_code == 422
    # The length rule, not the unknown-race rule, must be what rejected it
    assert response.json()["detail"][0]["type"] == "string_too_long"


def test_prediction_failure_hides_internals(client, monkeypatch):
    # Stand-in model whose failure message mimics a library leaking internals
    class ExplodingModel:
        # Same interface predict() calls on real models
        def predict(self, features):
            # Text that must never reach the client
            raise RuntimeError("secret-internal")

    # setitem on the shared artifact dict; monkeypatch restores the real model after the test
    monkeypatch.setitem(ml_models["qatar"], "model", ExplodingModel())
    # Valid qatar input, so the request reaches the inference try-block
    response = client.post("/predict", json=payload("qatar", 82.2, 91.1, 25.0))
    # Model failure is a server-side fault
    assert response.status_code == 500
    # Client sees only the generic message
    assert response.json()["detail"] == "Prediction failed"
    # And nothing from the exception text anywhere in the body
    assert "secret-internal" not in response.text


def test_shutdown_clears_lookup_data(client):
    # Snapshot shared state: a second lifespan below clears the dicts the module client also uses
    saved_models, saved_lookup = dict(ml_models), dict(lookup_data)
    # try/finally restores state even if an assert fails, so test order stays irrelevant
    try:
        # A separate context runs one full startup + shutdown cycle
        with TestClient(app):
            # Startup must have filled the table, or the post-shutdown check proves nothing
            assert "data" in lookup_data
        # Shutdown must drop the lookup table, not only the models
        assert lookup_data == {}
        # Models are cleared on the same shutdown path
        assert ml_models == {}
    finally:
        # Put the module client's models back for later tests
        ml_models.update(saved_models)
        # And its lookup table
        lookup_data.update(saved_lookup)


def test_reordered_features_rejected_at_load():
    # Swap the first two columns: same names, wrong order, exactly the silent-wrong-pace case
    reordered = list(EXPECTED_FEATURES["usa"])
    # In-place swap keeps the set of names identical so only order differs
    reordered[0], reordered[1] = reordered[1], reordered[0]
    # Fake bundle shaped like the real usa pickle; no model file on disk is touched
    fake = {"model": object(), "imputer": None, "features": reordered}
    # Load check must refuse it so the race is skipped instead of served
    assert feature_names_match("usa", fake) is False
    # Control: the correct order passes, so the rejection above is about order, not the fake
    assert feature_names_match("usa", {**fake, "features": list(EXPECTED_FEATURES["usa"])}) is True


def test_predict_emits_no_feature_name_warning(client):
    # record=True captures every warning raised inside the block instead of printing it
    with warnings.catch_warnings(record=True) as caught:
        # "always" defeats the once-per-location registry, so a repeat warning cannot hide
        warnings.simplefilter("always")
        # usa runs both the imputer and the GBR, the two estimators that used to warn
        response = client.post("/predict", json=payload("usa", 94.5, 100.2, 35.0))
    # Request must succeed, or the absence of warnings proves nothing
    assert response.status_code == 200
    # The numpy-row warning must be suppressed at inference
    assert not [w for w in caught if "valid feature names" in str(w.message)]
