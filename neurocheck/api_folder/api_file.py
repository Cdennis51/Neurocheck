"""
NeuroCheck Backend API
======================

This module implements a FastAPI backend for EEG fatigue prediction and Alzheimer's MRI classification.

Key Features:
-------------
1. **Health Check Endpoint (`/health`)**
   - Returns backend status, version, and availability of preprocessing/model components.

2. **Debug Endpoint (`/debug`)**
   - Returns detailed system status and error information for troubleshooting.

3. **EEG Prediction Endpoint (`/predict/eeg`)**
   - Accepts EEG data as CSV.
   - Preprocesses EEG data before prediction when available.
   - Predicts fatigue level using a loaded ML model.
   - Falls back to dummy predictions with helpful error messages when components fail.

4. **Alzheimer's MRI Prediction Endpoint (`/predict/alzheimers`)**
   - Accepts MRI image files (JPEG/PNG).
   - Returns prediction with confidence score and attention map overlay.

5. **CORS Middleware**
   - Allows independent frontend-backend communication.
   - Configured for Streamlit frontend integration.

6. **Model & Preprocessing Management**
   - Dynamically imports preprocessing and model components with graceful failure handling.
   - Logs detailed warnings for debugging component failures.
   - Supports "development" mode with informative dummy responses.


Development Notes:
------------------
- Use `/debug` endpoint to troubleshoot component loading issues.
- In production mode, ensure preprocessing and model modules are installed.
- In development mode, dummy predictions include error details for debugging.

Run Locally:
------------
    uvicorn neurocheck.api_folder.api_file:app --host 0.0.0.0 --port 8000 --reload
"""
import sys
import os
from io import BytesIO
import logging
import random
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from xgboost import XGBClassifier

# Set path before imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml_logic'))

# Import Alzheimer's Modules
from alzheimers.alzheimers_model import predict as predict_alzheimers_image
from alzheimers.alzheimers_preprocess import resize_upload as resize_alzheimers_upload

# === Configure Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# === Dummy Response Generator ===
def create_dummy_response(user_document: str, mode: str):
    """Generate dummy response for development/testing"""
    dummy_classes = ["0*", "1*"]
    return {
        "backend_status": f"development_mode_{mode}",
        "fatigue_class": random.choice(dummy_classes),
        "confidence": round(random.uniform(0.7, 0.95), 2),
        "filename": user_document,
        "note": "It appears our analysis tools are offline.\n\
            We're sorry about that. We're fixing it now.\n\
            We should be back online and happy to help soon."
        }

# === Require Named File for User Input===
def require_filename(received_file) -> str:
    """Ensures file and filename are present, returns lowercase name or raises ValueError."""
    if not received_file or not received_file.filename:
        raise ValueError("Expected a file with a valid filename")
    return received_file.filename.lower()

# === Import Processing Function ===
# Attempt to import preprocessing components
PREPROCESS_EEG_AVAILABLE = False
PREPROCESS_EEG_IMPORT_ERROR = None
PREPROCESS_EEG = None
try:
    from neurocheck.ml_logic.preprocess import preprocess_eeg_df as PREPROCESS_EEG
    PREPROCESS_EEG_AVAILABLE = True
# If it fails, set to False and log warning.
# This allows the backend to run even if preprocessing is not available.
except ImportError as e:
    logging.warning("Preprocessor function load failed: %s", e)
    PREPROCESS_EEG_IMPORT_ERROR = str(e)


# === Model Module Import ===
# Attempt to import model components that will be instantiated and called later
MODEL_EEG_AVAILABLE = False
ML_LOAD_EEG_MODEL = None
MODEL_EEG_IMPORT_ERROR = None
try:
    from neurocheck.ml_logic.registry import retrieve_model as ml_load_eeg_model
    ML_LOAD_EEG_MODEL = ml_load_eeg_model
    MODEL_EEG_AVAILABLE = True
except ImportError as e:
    logging.warning("Model import function failed: %s", e)
    MODEL_EEG_IMPORT_ERROR = str(e)


# === Load Model At Startup ===
MODEL_EEG_LOADED = False
MODEL_EEG_RUN = None
MODEL_LOADING_ERROR = None
if MODEL_EEG_AVAILABLE:
    try:
        MODEL_EEG_RUN = ML_LOAD_EEG_MODEL(stage="Production")
        if MODEL_EEG_RUN is None:
            raise ValueError("Model loader returned None")
        else:
            logging.info("Model loaded successfully")
            MODEL_EEG_LOADED = True
    except (FileNotFoundError, IOError, ValueError, RuntimeError) as e:
        logging.warning("Failed to load model: %s", e)
        MODEL_LOADING_ERROR = str(e)

# === Initialize FastAPI App ===
app = FastAPI(
    title="NeuroCheck Backend",
    description="Neurocheck Backend for EEG and Alzheimers",
    version="0.5.0"
)

# === CORS Functionality for Independent Front End, Back End Communication ===
# https://fastapi.tiangolo.com/tutorial/cors/#use-corsmiddleware

# Define Sources Assigned Full BackEnd API Calling Capabilities
# Revisit removing localhost capabilities after successful debugged deployment
ALLOWED_ORIGINS = [
    "http://localhost:8501",  # local Streamlit dev
    "https://neurocheck-frontend.streamlit.app", # Deployed Streamlit App
    "*"  # Temporary To Be Removed After Deployment
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Health Check Endpoint, Check If Backend Online===
@app.get("/health")
def health_check():
    """
    Health check endpoint for the NeuroCheck backend.

    Returns:
        dict: A JSON response containing:
        - status: Service availability ("online", "degraded", "offline")
        - version: Current API version
        - message: User-friendly status message
        - components: Detailed component availability and operational mode
        - endpoints: Available API endpoints for discovery
    """
    if PREPROCESS_EEG_AVAILABLE and MODEL_EEG_LOADED:
        service_status = "online"
        mode = "production"
    elif MODEL_EEG_LOADED or PREPROCESS_EEG_AVAILABLE:
        service_status = "degraded"
        mode = "partial"
    else:
        service_status = "degraded"
        mode = "development"
    health_status = {
        "status": service_status,
        "version": "0.4.2",
        "message": "Backend is running! Use /predict/eeg for predictions.",
        "components": {
            "preprocessing": PREPROCESS_EEG_AVAILABLE,
            "model": MODEL_EEG_LOADED,
            "mode": mode if (PREPROCESS_EEG_AVAILABLE and MODEL_EEG_LOADED) else f"{mode}: See '/debug'"
        },
        "endpoints": ["/health", "/debug", "/predict/eeg"]
    }
    return health_status

# === Debug Endpoint ===
@app.get("/debug")
def debug_status():
    """
    Returns system status information for debugging.

    Provides current state of model loading, preprocessing availability,
    and detailed error messages for troubleshooting component failures.
    """
    errors = {}

    if PREPROCESS_EEG_IMPORT_ERROR:
        errors["preprocess_import"] = PREPROCESS_EEG_IMPORT_ERROR
    if MODEL_EEG_IMPORT_ERROR:
        errors["model_import"] = MODEL_EEG_IMPORT_ERROR
    if MODEL_LOADING_ERROR:
        errors["model_loading"] = MODEL_LOADING_ERROR
    return {
        "MODEL_AVAILABLE": MODEL_EEG_AVAILABLE,
        "MODEL_LOADED": MODEL_EEG_LOADED,
        "PREPROCESS_AVAILABLE": PREPROCESS_EEG_AVAILABLE,
        "errors": errors if errors else "No errors: All components loaded successfully"
    }

# === EEG Prediction Endpoint ===
@app.post("/predict/eeg")
async def predict_eeg(eeg_file: UploadFile = File(...)):
    """
    EEG prediction endpoint with comprehensive error handling and fallback responses.

    Accepts:
        - CSV file uploads (EDF support planned for future)

    Process:
        - Reads uploaded EEG file → DataFrame with column cleaning
        - Preprocesses EEG data when preprocessing module is available
        - Predicts fatigue level using loaded XGBoost model with custom threshold (0.45)
        - Falls back to informative dummy responses when components fail

    Returns:
        dict: JSON response containing:
            - backend_status: "Production" or "development_mode_[error_type]"
            - fatigue_class: "0" (not fatigued) or "1" (fatigued), or dummy class
            - confidence: Model confidence score (0-1) or dummy confidence
            - filename: Original uploaded filename
            - preprocessing_used: Boolean indicating if preprocessing succeeded
            - note: Error explanation (in dummy responses only)
    """
    user_eeg_file_name = require_filename(eeg_file)


    # Step 1: Load EEG into DataFrame
    try:
        if user_eeg_file_name.endswith(".csv"):
            contents = await eeg_file.read()
            eeg_df = pd.read_csv(BytesIO(contents))
            eeg_df.columns = eeg_df.columns.str.strip()
        # elif user_eeg_file_name.endswith(".edf"):
        #     eeg_df = read_edf_to_dataframe(eeg_file.file)  # Not yet implemented
        else:
            logging.warning("Unsupported file format: %s", user_eeg_file_name)
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload CSV."
        )

    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError, IOError) as e:
        logging.error("Failed to read EEG file: %s", e)
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read EEG file: {str(e)}"
        ) from e


    # Step 2: Try preprocessing
    preprocessing_success = False
    proc_eeg_df = eeg_df

    if PREPROCESS_EEG_AVAILABLE and PREPROCESS_EEG:
        try:
            proc_eeg_df = PREPROCESS_EEG(eeg_df)
            logging.info("Preprocessing completed successfully")
            preprocessing_success = True
        except (ValueError, RuntimeError) as e:
            logging.warning("Preprocessing failed: %s", e)

    # Step 3: Try predicting
    if MODEL_EEG_LOADED and MODEL_EEG_RUN:
        try:
            proc_eeg_df.columns = proc_eeg_df.columns.str.strip()
            logging.info("Using columns for prediction: %s", proc_eeg_df.columns.tolist())
            proba = float(MODEL_EEG_RUN.predict_proba(proc_eeg_df)[0, 1])
            prediction = int(proba > 0.45)  # apply custom threshold 0.45 probability >0.45 you get fatigued below - 0 not fatigued.
            result = {
                "backend_status": "Production",
                "fatigue_class": str(prediction),
                "confidence": round(proba if prediction == 1 else 1 - proba, 4),
                "filename": user_eeg_file_name,
                "preprocessing_used": preprocessing_success
            }

        except (ValueError, RuntimeError, KeyError) as e:
            logging.warning("Prediction failed: %s", e)
            result = create_dummy_response(user_eeg_file_name, f"prediction_error: {str(e)}")

    else:
        result = create_dummy_response(user_eeg_file_name, "Development")

    return result

# === Local Dev Server Runner ===
if __name__ == "__main__":
    uvicorn.run("neurocheck.api_folder.api_file:app", host="0.0.0.0", port=8000, reload=True)
