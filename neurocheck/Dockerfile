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




# === Step 01: Build Base Image ===
# FROM python:3.10-slim AS base

# # Don't write .pyc files and ensure logs are printed immediately
# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1

# # Set working directory
# WORKDIR /app

# # Install required system packages to support scientific libraries (NumPy, Pandas, etc.)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     gcc \
#     gfortran \
#     libatlas-base-dev \
#     && rm -rf /var/lib/apt/lists/*

# # === Step 02: Install Python Dependencies and Cache Model ===
# FROM base AS builder

# # Copy Python dependency list
# COPY requirements.txt .

# # Install Python packages and pre-cache the Hugging Face Alzheimer model
# RUN pip install --upgrade pip && \
#     pip install --prefix=/install -r requirements.txt && \
#     python -c "from transformers import pipeline; pipeline('image-classification', model='DHEIVER/Alzheimer-MRI')"

# # === Step 03: Build Minimal Runtime Image ===
# FROM base

# # Copy Python dependencies (from builder stage) into this final stage
# COPY --from=builder /install /usr/local

# # Copy the cached Alzheimer model so it can run offline
# COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# # Copy your application source code (FastAPI app, ML code, etc.)
# COPY . /app/neurocheck

# # Optional: silence Hugging Face warning messages
# ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# # Expose port 8080 (used by FastAPI/Uvicorn)
# EXPOSE 8080

# # Run the FastAPI app using Uvicorn (adjust filename if needed)
# CMD ["uvicorn", "neurocheck.api_folder.api_file_alzheimers:app", "--host", "0.0.0.0", "--port", "8080"]

#
