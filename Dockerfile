# ─── News Recommendation Engine — EC2 Dockerfile ─────────────────────────────
# Multi-stage build: keeps the final image lean (no build tools).
# Base: python:3.11-slim
#
# Build & push:
#   docker build -t news-rec:latest .
#   docker tag  news-rec:latest <your-ecr-uri>/news-rec:latest
#   docker push <your-ecr-uri>/news-rec:latest
#
# Run directly (no docker-compose):
#   docker run --env-file .env -p 8000:8000 news-rec:latest

FROM python:3.11-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Build stage ──────────────────────────────────────────────────────────────
FROM base AS builder

WORKDIR /build
COPY app/requirements.txt /build/requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r /build/requirements.txt

# Install boto3 / pyathena (needed for Athena ingestion)
RUN pip install --no-cache-dir --prefix=/install \
        boto3>=1.34.0 \
        pyathena>=3.3.0 \
        sqlalchemy>=2.0.0

# ── Final stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS final

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy project files
COPY . /app

# Create directories for persistent data / model cache
RUN mkdir -p \
        /app/local_store \
        /app/qdrant_storage \
        /root/.cache/huggingface

# ── Environment ───────────────────────────────────────────────────────────────
# These are defaults; override via --env-file or ECS task definitions.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_NO_TF=1 \
    TOKENIZERS_PARALLELISM=false \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command — run FastAPI with uvicorn
CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
