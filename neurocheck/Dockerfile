# === Step 01: Base Image ===
FROM python:3.10-slim AS base

# Environment variables: no pyc files, unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies needed for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    gfortran \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*


# === Step 02: Builder Stage (optional) ===
# You can still keep this to cache build dependencies if you want, but not mandatory

FROM base AS builder

# Copy requirements.txt first (for cache)
COPY requirements.txt /app/

RUN pip install --upgrade pip
RUN pip uninstall -y torch torchvision torchaudio || true
RUN pip install -r requirements.txt

# === Step 03: Final runtime image ===
FROM base

WORKDIR /app

# Copy requirements.txt to install dependencies here
COPY requirements.txt /app/

# Install Python dependencies directly in final image
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy your backend source code
COPY neurocheck /app/
ENV PYTHONPATH=/app:$PYTHONPATH

# Create Hugging Face cache directory and environment variables
RUN mkdir -p /app/cache/huggingface
ENV HF_HOME=/app/cache/huggingface
ENV TRANSFORMERS_CACHE=/app/cache/huggingface

EXPOSE 8080

CMD ["uvicorn", "neurocheck.api_folder.api_file:app", "--host", "0.0.0.0", "--port", "8080"]
