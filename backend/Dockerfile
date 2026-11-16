# Build with repo root as context: `docker compose build backend` (see docker-compose.yml)
# or manually: `docker build -f backend/Dockerfile -t meetmind-backend .` from the repo root.
FROM python:3.11-slim

# libpq-dev/gcc: needed if you switch DATABASE_URL to Postgres (psycopg2 from source).
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ai-pipeline/ /app/ai-pipeline/
COPY backend/ /app/backend/

# WORKDIR matters here: backend/requirements.txt has "-e ../ai-pipeline", which
# resolves relative to the current directory pip is run from.
WORKDIR /app/backend
RUN pip install --no-cache-dir -r requirements.txt

# Install CPU-only PyTorch
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.8.0+cpu torchaudio==2.8.0+cpu


# Install WhisperX and compatible dependencies while preserving CPU-only PyTorch
RUN printf "torch==2.8.0+cpu\ntorchaudio==2.8.0+cpu\ntorchvision==0.23.0+cpu\nhuggingface-hub<1.0.0\n" > /tmp/whisperx-constraints.txt \
    && pip install --no-cache-dir \
       --index-url https://pypi.org/simple \
       --extra-index-url https://download.pytorch.org/whl/cpu \
       -c /tmp/whisperx-constraints.txt \
       whisperx==3.8.6 datasets==4.3.0


EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
