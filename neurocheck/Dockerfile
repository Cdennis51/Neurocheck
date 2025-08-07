# === Step 1: Base image with common setup ===
FROM python:3.10-slim AS base

# Avoid .pyc files, enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app
#WORKDIR neurocheck /app/

# Install system-level dependencies required by Python libs (e.g. OpenCV, GradCAM)
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  gcc \
  gfortran \
  libatlas-base-dev \
  libgl1-mesa-glx \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  && rm -rf /var/lib/apt/lists/*

# === Step 2: Builder to install Python dependencies ===
FROM base AS builder

# Copy requirements early for caching
COPY requirements.txt .

# Install all dependencies into /install
RUN pip install --upgrade pip && \
    pip install --prefix=/install -r requirements.txt

# === Step 3: Final runtime image ===
FROM base

# Optional: set model cache directory (avoids HF warning)
ENV HF_HOME=/app/hf_cache

# Copy Python site-packages from builder
COPY --from=builder /install /usr/local

# Copy application code into container
COPY . /app/neurocheck

# Set working directory for runtime
WORKDIR /app

# (Optional) Preload model into container during build
RUN python -c "from transformers import pipeline; pipeline('image-classification', model='DHEIVER/Alzheimer-MRI')"

# Expose FastAPI port (comment this out for GCP)
# EXPOSE 8080

# Start FastAPI app (adjust filename if different)
# Start FastAPI app with uvicorn
# CMD ["uvicorn", "neurocheck.api_folder.api_file:app", "--host", "0.0.0.0", "--port", ${PORT}] #old version
CMD exec uvicorn neurocheck.api_folder.api_file:app --host 0.0.0.0 --port ${PORT}
