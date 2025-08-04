# === Step 01: Build Base Image ===
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


# === Step 02: Install Python Dependencies (Builder Stage) ===
FROM base AS builder

# Copy requirements first for caching layers
COPY requirements.txt .

# Upgrade pip and install dependencies to /install folder
RUN pip install --upgrade pip && \
    pip install --prefix=/install -r requirements.txt

# === Step 03: Final Runtime Image ===
FROM base

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy backend source code
COPY . /app/neurocheck

# Create Hugging Face cache directory
RUN mkdir -p /app/cache/huggingface

# Set environment variable to use cache folder for transformers
ENV TRANSFORMERS_CACHE=/app/cache/huggingface

# Preload Alzheimer model to cache during build (speeds up runtime)
RUN python -c "from transformers import pipeline; pipeline('image-classification', model='DHEIVER/Alzheimer-MRI')"

# Expose the port FastAPI listens on
EXPOSE 8080

# Start FastAPI app with uvicorn
CMD ["uvicorn", "neurocheck.api_folder.api_file:app", "--host", "0.0.0.0", "--port", "8080"]
