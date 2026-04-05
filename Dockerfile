# ── Stage 1: dependency builder ──────────────────────────────────────────────
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# System deps needed only for compilation – not carried into final image
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: minimal runtime image ───────────────────────────────────────────
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    # Keep malloc lean inside the 8 GB HF Space limit
    MALLOC_TRIM_THRESHOLD_=65536 \
    # OpenEnv runtime flag
    OPENENV_ENV_MODULE="env" \
    OPENENV_ENTRY_POINT="make_env"

# Non-root user required by Hugging Face Spaces
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Application source
COPY env.py inference.py openenv.yaml ./

# HF Spaces default port
EXPOSE 7860

USER appuser

# Validate the OpenEnv spec on startup, then launch inference
CMD ["sh", "-c", "openenv validate openenv.yaml && python inference.py"]
