"""
NeuroCheck Backend API
======================

This module implements a FastAPI backend for EEG fatigue prediction.

Key Features:
-------------
1. **Health Check Endpoint (`/health`)**
   - Returns backend status, version, and availability of preprocessing/model components.

2. **EEG Prediction Endpoint (`/predict/eeg`)**
   - Accepts EEG data as CSV (EDF support planned for future).
   - (Optional) Preprocesses EEG data before prediction.
   - Predicts fatigue level using a loaded ML model.
   - Falls back to dummy predictions if preprocessing or model is unavailable.

3. **CORS Middleware**
   - Allows independent frontend-backend communication.
   - Default: all origins allowed (configure for production).

4. **Model & Preprocessing Management**
   - Tries to import preprocessing and model components dynamically.
   - Logs warnings if unavailable to allow backend to run in development mode.
   - Supports "development" mode with dummy responses when real model is missing.

5. **EDF Support (Future)**
   - Placeholder `read_edf_to_dataframe` for future EDF file integration.

Development Notes:
------------------
- In production mode, ensure preprocessing and model modules are installed.
- In development mode, dummy predictions will be generated for testing.

Run Locally:
------------
    uvicorn neurocheck.api_file:app --host 0.0.0.0 --port 8000 --reload
"""
from io import BytesIO
import logging
import random
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from xgboost import XGBClassifier

# === Configure Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


# === EDF File Support (Post MVP) ===
## EDF file support requires mne package

## Attempt to import mne for EDF file support
# try:
#     import mne  # for EDF files
#     EDF_SUPPORT = True

# except ImportError:
#     EDF_SUPPORT = False


# === EDF Reader Function (Post MVP) ===
def read_edf_to_dataframe(file_obj):
    """Stub for EDF file parsing. Requires mne to implement."""
    raise NotImplementedError("EDF file support not yet implemented. Use CSV instead.")


# === Preprocessing Module Import ===
# Attempt to import preprocessing components
try:
    from neurocheck.ml_logic.preprocess import preprocess_eeg_df as preprocess_eeg
    PREPROCESS_AVAILABLE = True

# If it fails, set to False and log warning.
# This allows the backend to run even if preprocessing is not available.
except ImportError as e:
    logging.warning("Preprocessing not available: %s", e)
    preprocess_eeg = None
    PREPROCESS_AVAILABLE = False


# === Model Module Import ===
# Attempt to import model components that will be instantiated and called later
try:
    from neurocheck.ml_logic.registry import retrieve_model as ml_load_eeg_model
    MODEL_AVAILABLE = True
except ImportError as e:
    logging.warning("Model components not available: %s", e)
    ml_load_eeg_model = None
    MODEL_AVAILABLE = False

if MODEL_AVAILABLE:
    try:
        eeg_model = ml_load_eeg_model(stage="Production")
        if eeg_model is None:
            logging.warning("Model loaded but is None.")
            MODEL_LOADED = False
        else:
            MODEL_LOADED = True
            logging.info("Model loaded successfully")
    except (FileNotFoundError, IOError, ValueError, RuntimeError) as e:
        logging.warning("Failed to load model: %s", e)
        eeg_model = None
        MODEL_LOADED = False

# === Initialize FastAPI App ===
app = FastAPI(
    title="NeuroCheck Backend",
    description="EEG Fatigue Prediction Backend",
    version="0.4.0"
)


# === CORS Functionality for Independent Front End, Back End Communication ===
# https://fastapi.tiangolo.com/tutorial/cors/#use-corsmiddleware

# Define Sources Assigned Full BackEnd API Calling Capabilities
# Revisit removing localhost capabilities after successful debugged deployment
ALLOWED_ORIGINS = [
    "http://localhost:8501",  # local Streamlit dev
    "https://neurocheck-frontend.streamlit.app/", # Deployed Streamlit App
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Global Model Placeholder ===
#MODEL_LOADED = False
#eeg_model = None


# === Load Model At Startup===
# Attempt to load the model if available
"""if MODEL_AVAILABLE:

    try:
        eeg_model = ml_load_eeg_model() # Will be None during development
        MODEL_LOADED = True
        logging.info("Model loaded successfully")


    except (FileNotFoundError, IOError) as e:
        logging.warning("Warning: Failed to load model: %s", e)
        MODEL_LOADED = False """


# === Health Check Endpoint, Check If Backend Online===
@app.get("/health")
def index():
    """
    Health check endpoint for the NeuroCheck backend.

    Returns:
        dict: A JSON response confirming:
        - backend status
        - version info
        - availability of preprocessing/model
        - current mode (development vs production)
    """
    health_status = {
        "status": "online",
        "version": "0.1.0",
        "message": "Backend is running! Use /predict/eeg for predictions.",
        "components": {
            "preprocessing": PREPROCESS_AVAILABLE,
            "model": MODEL_LOADED,
            "mode": "production" if (PREPROCESS_AVAILABLE and MODEL_LOADED) else "development"
        }
    }
    return health_status


# === EEG Prediction Endpoint ===
@app.post("/predict/eeg")
async def predict_eeg(file: UploadFile = File(...)):
    """
    EEG prediction endpoint with fallback dummy responses.

    Accepts:
        - CSV (and EDF in future) file uploads

    Process:
        - Reads uploaded EEG file → DataFrame
        - (Future) Preprocess data before prediction
        - (Future) Predicts fatigue level using ML model

    Returns:
        dict: JSON with fatigue_class + confidence (dummy if no model)
    """
    filename = file.filename.lower()

    # Step 1: Load EEG into DataFrame
    try:
        if filename.endswith(".csv"):
            contents = await file.read()
            eeg_df = pd.read_csv(BytesIO(contents))
            eeg_df.columns = eeg_df.columns.str.strip()
        elif filename.endswith(".edf"):
            eeg_df = read_edf_to_dataframe(file.file)  # Not yet implemented
        else:
            logging.warning("Unsupported file format: %s", filename)
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload CSV (EDF coming soon)."
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

    if PREPROCESS_AVAILABLE and preprocess_eeg:
        try:
            proc_eeg_df = preprocess_eeg(eeg_df)
            print("Preprocessing done")
            preprocessing_success = True

        except (ValueError, RuntimeError) as e:
            logging.warning("Preprocessing failed: %s", e)
            #proc_eeg_df = eeg_df  # fallback to raw data

    # === Dummy Response Generator ===
    def create_dummy_response(filename: str, mode: str):
        """Generate dummy response for development/testing"""
        dummy_classes = ["fatigued", "not fatigued"]

        return {
            "backend_status": f"development_mode_{mode}",
            "fatigue_class": random.choice(dummy_classes),
            "confidence": round(random.uniform(0.7, 0.95), 2),
            "filename": filename,
            "note": f"Dummy response - {mode}. Real model integration pending."
            }

    # Step 3: Try predicting
    if MODEL_LOADED and eeg_model:
        try:
            proc_eeg_df.columns = proc_eeg_df.columns.str.strip()
            proba = eeg_model.predict_proba(proc_eeg_df)[0, 1] #
            prediction = int(proba > 0.45)  # apply custom threshold 0.45 probability >0.45 you get fatigued below - 0 not fatigued.
            result = {
                "backend_status": "production",
                "fatigue_class": str(prediction),
                "confidence": round(proba if prediction == 1 else 1 - proba, 4), #Calculate's the confidence in the predicted class, not just fatigued.
                "filename": filename,
                "preprocessing_used": preprocessing_success
            }



        except (ValueError, RuntimeError, KeyError) as e:
            logging.warning("Prediction failed: %s", e)
            result = None
            #result = create_dummy_response(file.filename, "prediction_error")

    else:
        result = None
        #result = create_dummy_response(file.filename, "development")

    return result

# === Local Dev Server Runner ===
if __name__ == "__main__":
    uvicorn.run("neurocheck.api_folder.api_file:app", host="0.0.0.0", port=8000, reload=True)
