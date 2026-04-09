# ── SENTINEL-PR Dockerfile ────────────────────────────────────────────────────

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install all deps — numpy required by env.py at import time
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

# ── Copy ALL required files including grader.py ───────────────────────────────
COPY env.py inference.py app.py grader.py openenv.yaml requirements.txt ./

# Copy server package
COPY server/ ./server/

# Build-time checks
RUN test -f /app/openenv.yaml \
    && echo "[OK] openenv.yaml present" \
    || (echo "[FAIL] openenv.yaml missing" && exit 1)

RUN test -f /app/grader.py \
    && echo "[OK] grader.py present" \
    || (echo "[FAIL] grader.py missing" && exit 1)

# Verify env.py and grader.py import cleanly — catches missing deps at build time
RUN python -c "import env; print('[OK] env.py imports')"
RUN python -c "import grader; print('[OK] grader.py imports')"

EXPOSE 7860

USER appuser

# Starts the long-running FastAPI+Gradio server.
# Container stays ALIVE so validator can reach /reset and /step on port 7860.
CMD ["python", "app.py"]
