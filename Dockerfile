# ── SENTINEL-PR Dockerfile ────────────────────────────────────────────────────
# ROOT CAUSE of "Error: Path is not a directory: /Meta/openenv.yaml":
#
#   The openenv CLI's validate command expects:
#     openenv validate <DIRECTORY>   ← directory that CONTAINS openenv.yaml
#   Not:
#     openenv validate <FILE>        ← causes "Path is not a directory" error
#
#   Confirmed by screenshot:
#     File exists: -rwxrwxrwx 1 root root 9990 /Meta/openenv.yaml  ✓
#     Error comes from passing the file path to the CLI, not the dir.
#
#   Fix: always pass the directory (/app), never the file (/app/openenv.yaml)
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt


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

# Copy application files – openenv.yaml is copied as a regular FILE into /app/
COPY env.py inference.py openenv.yaml ./

# Build-time sanity check: openenv.yaml must be a regular file
RUN test -f /app/openenv.yaml \
    && echo "[BUILD OK] $(ls -la /app/openenv.yaml)" \
    || (echo "[BUILD FAIL] openenv.yaml is missing or not a file" && exit 1)

EXPOSE 7860

USER appuser

# ── THE FIX: pass DIRECTORY to openenv validate, not file path ────────────────
#
# WRONG (causes screenshot error):  openenv validate /app/openenv.yaml
# CORRECT:                          openenv validate /app
#
# The CLI looks for openenv.yaml INSIDE the directory you pass.
# Passing the yaml file path directly gives "Path is not a directory".
CMD ["sh", "-c", "\
  echo '[INFO] Working directory contents:' && ls -la /app/; \
  echo '[INFO] openenv.yaml:' && ls -la /app/openenv.yaml; \
  echo '[INFO] Validating: openenv validate /app'; \
  openenv validate /app \
  && echo '[INFO] Validation passed. Starting inference...' \
  && python inference.py \
"]
