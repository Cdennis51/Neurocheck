# === Step 01: Build Base Image ===
FROM python:3.10-slim AS base

# Instruct Python to Not Write pyc, enable unbuffered logs for debugging
## PYTHONDONTWRITEBYTECODE=1: Don't create .pyc cache files (cleaner container)
## PYTHONUNBUFFERED=1: Print Python output immediately

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install compilers needed to build numpy, pandas, xgboost, scikit-learn
## rm -rf /var/lib/apt/lists/*: Cleans up package cache to save space

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    gfortran \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*


# === Step 02: Install Python Dependencies===
# Build an intermediate staging area (builder) to install all packages to be discarded at completion
FROM base AS builder

# First copy requirements
COPY requirements.txt .

# Install dependencies into a temporary folder (/install) to be discarded at completion
RUN pip install --upgrade pip && \
    pip install --prefix=/install -r requirements.txt

# === Step 03: Build Runtime Image without Keeping "Construction Tools"===
# Prepare to make a new staging area for final deployment
FROM base

# Duplicate installed packages from intermediate staging area without gcc, build tools, or pip cache
COPY --from=builder /install /usr/local

# Copy only backend source code
COPY . /app/neurocheck

# Expose FastAPI port
EXPOSE 8080

# Command to start FastAPI app with Uvicorn
CMD ["uvicorn", "neurocheck.api_folder.api_file:app", "--host", "0.0.0.0", "--port", "8080"]




# # === Step 01: Build Base Image ===
# FROM python:3.10-slim AS base

# # Environment variables to improve Python behavior
# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1 \
#     TRANSFORMERS_CACHE=/app/cache/huggingface

# # Set working directory
# WORKDIR /app

# # Install system dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     gcc \
#     gfortran \
#     libatlas-base-dev \
#     libgl1 \
#     && rm -rf /var/lib/apt/lists/*

# # === Step 02: Install Python Dependencies in a Builder Stage ===
# FROM base AS builder

# COPY requirements.txt .

# # Install dependencies into /install
# RUN pip install --upgrade pip && \
#     pip install --prefix=/install -r requirements.txt

# # === Step 03: Final Runtime Image ===
# FROM base

# # Copy installed packages from builder
# COPY --from=builder /install /usr/local

# # Copy backend code
# COPY . /app/neurocheck

# # Create Hugging Face model cache folder
# RUN mkdir -p /app/cache/huggingface

# # Preload Alzheimer model to cache (avoids runtime download)
# RUN python -c "from transformers import pipeline; pipeline('image-classification', model='DHEIVER/Alzheimer-MRI')"

# # Expose FastAPI port
# EXPOSE 8080

# # Start FastAPI backend
# CMD ["uvicorn", "neurocheck.api_folder.api_file:app", "--host", "0.0.0.0", "--port", "8080"]
