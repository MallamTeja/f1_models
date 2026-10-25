"""Runtime settings for the F1 pace API, read once from the environment.

Env vars (all optional):
    PORT        - port uvicorn binds to; read by the Docker/Render start command, default 8000.
    MODELS_DIR  - folder holding the *.joblib models and lookup_data.json; default <repo>/models.
"""

# os gives env lookups so deploy targets can override paths without code edits
import os

# Path keeps directory math portable across Windows dev boxes and Linux containers
from pathlib import Path

# Anchor defaults to this file, not the CWD, so `uvicorn main:app` works from any folder
REPO_ROOT = Path(__file__).resolve().parent

# Models can live outside the image (mounted volume) on hosts that set MODELS_DIR
MODELS_DIR = Path(os.environ.get("MODELS_DIR", REPO_ROOT / "models"))

# One source for the version so FastAPI's OpenAPI doc and /info cannot drift apart
APP_VERSION = "1.3.0"
