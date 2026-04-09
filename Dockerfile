# ── SENTINEL-PR Dockerfile ────────────────────────────────────────────────────

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# numpy needs gcc; gradio needs build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install ALL deps explicitly — numpy is required by env.py at import time
RUN pip install --prefix=/install --no-cache-dir \
        numpy \
        "pydantic>=2.9.0" \
        "bandit>=1.7.0" \
        "safety>=3.2.0" \
        "openai>=2.7.2" \
        "requests>=2.32.0" \
        "urllib3>=2.2.0" \
        "flask>=3.0.3" \
        "gradio>=5.25.0" \
        "uvicorn>=0.29.0" \
        "fastapi>=0.111.0" \
        packaging \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt || true


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    MALLOC_TRIM_THRESHOLD_=65536 \
    OPENENV_ENV_MODULE="env" \
    OPENENV_ENTRY_POINT="make_env"

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

COPY --from=builder /install /usr/local

# Copy ALL application files
COPY env.py inference.py app.py grader.py openenv.yaml requirements.txt ./

# Copy server package
COPY server/ ./server/

# Build-time checks
RUN test -f /app/openenv.yaml \
    && echo "[BUILD OK] openenv.yaml present" \
    || (echo "[BUILD FAIL] openenv.yaml missing" && exit 1)

# Verify env.py imports cleanly — catches missing deps at BUILD time not runtime
RUN python -c "import env; print('[BUILD OK] env.py imports successfully')"

EXPOSE 7860

USER appuser

# KEY: Starts the long-running FastAPI+Gradio server so container stays ALIVE.
# Validator needs /reset and /step endpoints reachable on port 7860.
CMD ["python", "app.py"]
