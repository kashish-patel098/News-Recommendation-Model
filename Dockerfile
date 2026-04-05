# Backend Dockerfile
# Builds the FastAPI + BGE-m3 + PyTorch recommendation engine

FROM python:3.11-slim

# System dependencies needed by PyTorch / transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/         ./app/
COPY local_store/ ./local_store/
COPY scripts/     ./scripts/
COPY .env.example ./.env.example
COPY requirements.txt ./requirements.txt

# Create persistent directories
RUN mkdir -p local_store

# Suppress TF/Keras on startup
ENV TRANSFORMERS_NO_TF=1
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TOKENIZERS_PARALLELISM=false

# Pre-download BGE-m3 model weights into the image
# Comment this out if you want a smaller image and prefer runtime download
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
    AutoTokenizer.from_pretrained('BAAI/bge-m3'); \
    AutoModel.from_pretrained('BAAI/bge-m3')" \
 && echo "Model weights cached."

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
