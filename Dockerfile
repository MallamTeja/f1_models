# syntax=docker/dockerfile:1
# Why the syntax line: pins the Dockerfile frontend so BuildKit features behave the same on every builder (local, GHCR CI).

# Why an ARG: both stages must use the SAME base, or the venv's python symlink breaks when copied across.
# Why a patch-pinned tag: a floating "3.11-slim" silently changes under us; a pinned tag makes builds reproducible.
ARG PYTHON_IMAGE=python:3.11.14-slim-bookworm

# ---------- Stage 1: builder ----------
# Why a separate builder stage: pip caches and build tooling stay here and never reach the shipped image.
FROM ${PYTHON_IMAGE} AS builder

# Why: skip pip's network version check (noise + slower build) and keep no wheel cache in the layer.
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1

# Why a venv: gives one self-contained directory we can copy whole into the runtime stage.
RUN python -m venv /opt/venv

# Why put the venv first on PATH: every later pip/python call installs into and runs from the venv, not system Python.
ENV PATH="/opt/venv/bin:$PATH"

# Why copy requirements alone first: this layer (and the slow pip install) is cached until requirements.txt changes.
COPY requirements.txt .

# Why --no-cache-dir: wheel cache would only bloat this layer; all deps ship manylinux wheels so no compiler is needed.
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Stage 2: runtime ----------
# Why the same base: the copied venv points at /usr/local/bin/python, which must exist at the same path.
FROM ${PYTHON_IMAGE} AS runtime

# Why: links the GHCR package to its source repo so the package page shows the README and repo permissions apply.
LABEL org.opencontainers.image.source="https://github.com/mallamteja/f1_models"

# Why: no .pyc writes (read-only-friendly, smaller), unbuffered logs reach `docker logs` immediately, PORT default for local runs, venv on PATH so uvicorn resolves.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    PATH="/opt/venv/bin:$PATH"

# Why libgomp1 only: xgboost's OpenMP runtime needs it; removing apt lists keeps the layer from carrying the package index.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Why a fixed UID/GID 10001: never run as root, and a stable numeric ID lets orchestrators enforce runAsNonRoot and match volume ownership.
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

# Why: a fixed app directory keeps relative paths like models/ resolving the same as in the repo.
WORKDIR /app

# Why copy the venv instead of reinstalling: reuses the builder's resolved deps without shipping pip caches.
COPY --from=builder /opt/venv /opt/venv

# Why copy only these two files: the app is main.py + config.py; nothing else from the repo root is runtime code.
# Why no --chown: files stay root-owned, so the app user can read but never rewrite its own code or models.
COPY main.py config.py ./

# Why only .joblib + lookup_data.json: those are the models the API loads; .keras/.png are training artefacts and just add weight.
COPY models/*.joblib models/lookup_data.json ./models/

# Why switch user by numeric ID: Kubernetes runAsNonRoot can only verify a numeric UID, not a name.
USER 10001

# Why: documents the default listening port for `docker run -P` and readers; the real port still comes from $PORT.
EXPOSE 8000

# Why python urllib, not curl: slim has no curl; urlopen raises on 503 so the check fails until models are loaded. start-period covers model load time.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/health', timeout=4)" || exit 1

# Why sh -c: exec-form CMD does not expand ${PORT}. Why exec: uvicorn replaces the shell as PID 1 so it receives SIGTERM and shuts down gracefully.
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
