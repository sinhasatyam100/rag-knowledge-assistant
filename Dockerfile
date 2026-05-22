FROM python:3.10-slim

WORKDIR /app

# Install system build dependencies.
# build-essential: needed by some Python packages that compile C extensions.
# curl: useful for health check scripts and debugging inside the container.
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only build FIRST and separately.
# It is large (~200MB) and has its own index URL.
# Keeping it as a separate layer means it is only re-downloaded
# when this line changes — not on every requirements.txt update.
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.2.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install all other Python dependencies.
# --no-cache-dir keeps the image layer smaller.
RUN pip install --no-cache-dir -r requirements.txt

# ── Bake ML models into the image at build time ──────────────────────
# Without this, the first container startup downloads ~180MB of models,
# adding 6-12 seconds to cold start and requiring internet access at runtime.
# With this, models are available on local disk instantly.
# Tradeoff: image is ~700MB instead of ~200MB — acceptable for production.

# Embedding model — used for both ingestion and query-time embedding.
RUN python3 -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
print('Embedding model cached.')"

# Cross-encoder reranker — used in two-stage retrieval.
RUN python3 -c "\
from sentence_transformers import CrossEncoder; \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('Cross-encoder model cached.')"

# Copy application code LAST.
# Code changes most often — placing this after model downloads ensures
# a code change only invalidates the final layer, not the model cache.
COPY . .

# Create runtime directories expected by the application.
RUN mkdir -p logs uploads chroma_db

# Make the Celery worker entrypoint executable.
COPY worker_entrypoint.sh .
RUN chmod +x worker_entrypoint.sh

# Document the port the application listens on.
# Cloud Run routes traffic to port 8080 by default.
# docker-compose overrides PORT=8000 via environment variable.
EXPOSE 8080

# Start the FastAPI server.
# PORT env var lets Cloud Run (8080) and docker-compose (8000) both work
# without changing this line.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]