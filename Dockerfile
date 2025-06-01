FROM python:3.11-slim

# Set build arguments
ARG HF_TOKEN

# Set environment variables
ENV HF_TOKEN=${HF_TOKEN}
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache
ENV HF_DATASETS_CACHE=/app/model_cache
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_ENABLE_MPS_FALLBACK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    cmake \
    git \
    libpq-dev \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create model cache directory
RUN mkdir -p /app/model_cache

# Copy application code
COPY . .

# Build whisper.cpp
RUN cd app/lib/whisper && \
    cmake -B build -DWHISPER_METAL=OFF && \
    cmake --build build --config Release

# Create startup script
RUN echo '#!/bin/bash\n\
echo "Waiting for database..."\n\
while ! nc -z db 5432; do\n\
  sleep 0.1\n\
done\n\
echo "Database is up!"\n\
\n\
echo "Loading models..."\n\
python preload_models.py\n\
\n\
echo "Starting API server..."\n\
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload' > /app/start.sh && \
    chmod +x /app/start.sh

# Run the application
CMD ["/app/start.sh"] 