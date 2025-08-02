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

# Set path before module imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml_logic'))


# === Configure Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


# === Import EEG Processing Function ===
# Attempt to import EEG preprocessing components.
# If it fails, set to False and log warning.
PREPROCESS_EEG_AVAILABLE = False
PREPROCESS_EEG_IMPORT_ERROR = None
PREPROCESS_EEG = None
try:
    from neurocheck.ml_logic.preprocess import preprocess_eeg_df as PREPROCESS_EEG
    PREPROCESS_EEG_AVAILABLE = True
except ImportError as e:
    logging.warning("Preprocessor function load failed: %s", e)
    PREPROCESS_EEG_IMPORT_ERROR = str(e)


# === Import EEG Model Components ===
# Attempt to import model components that will be instantiated and called later.
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

# === Load EEG Model At Startup ===
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

# === Import Alzheimer's Modules ===
ALZHEIMERS_AVAILABLE = False
ALZHEIMERS_IMPORT_ERROR = None
PREDICT_ALZHEIMERS_IMAGE = None
RESIZE_ALZHEIMERS_UPLOAD = None

try:
    from neurocheck.ml_logic.alzheimers.alzheimers_model \
        import predict as predict_alzheimers_image
    from neurocheck.ml_logic.alzheimers.alzheimers_preprocess \
        import resize_upload as resize_alzheimers_upload

    PREDICT_ALZHEIMERS_IMAGE = predict_alzheimers_image
    RESIZE_ALZHEIMERS_UPLOAD = resize_alzheimers_upload
    ALZHEIMERS_AVAILABLE = True
    logging.info("Alzheimer's modules loaded successfully")
except ImportError as e:
    logging.warning("Alzheimer's modules load failed: %s", e)
    ALZHEIMERS_IMPORT_ERROR = str(e)

# === Helper Functions ===
def create_eeg_dummy_response(filename: str, error_type: str):
    """Generate consistent dummy response for EEG predictions"""
    return {
        "backend_status": f"development_mode_{error_type}",
        "fatigue_class": str(random.choice([0, 1])) + "*",
        "confidence": round(random.uniform(0.7, 0.95), 2),
        "filename": filename,
        "note": "Analysis tools offline. We're fixing it and should be back soon."
    }

def create_alzheimers_dummy_response(filename: str, error_type: str):
    """Generate consistent dummy response for Alzheimer's predictions"""
    dummy_classes = ["Normal", "Mild Cognitive Impairment", "Alzheimer's Disease"]
    return {
        "backend_status": f"development_mode_{error_type}",
        "prediction": random.choice(dummy_classes) + "*",
        "confidence": round(random.uniform(0.7, 0.95), 2),
        "filename": filename,
        "overlay": None,
        "note": "Analysis tools offline. We're fixing it and should be back soon."
    }

def require_filename(received_file) -> str:
    """Ensures file and filename are present, returns lowercase name or raises ValueError."""
    if not received_file or not received_file.filename:
        raise ValueError("Expected a file with a valid filename")
    return received_file.filename.lower()
# === Status Management ===
class ComponentStatus:
    """
    Centralized status manager for backend components and error tracking.

    Tracks availability and errors for all backend components (preprocessing,
    models, etc.) and provides standardized methods for determining service
    mode and error reporting.

    Attributes:
        components (dict): Component availability and error state tracking

    Methods:
        get_service_mode(): Returns 'production', 'partial', or 'development'
        get_errors(): Returns dict of components with active errors
    """
    def __init__(self):
        self.components = {
            'eeg_preprocessing': {
                'available': PREPROCESS_EEG_AVAILABLE,
                'error': PREPROCESS_EEG_IMPORT_ERROR
            },
            'eeg_model': {
                'available': MODEL_EEG_LOADED,
                'error': MODEL_EEG_IMPORT_ERROR or MODEL_LOADING_ERROR
            },
            'alzheimers': {
                'available': ALZHEIMERS_AVAILABLE,
                'error': ALZHEIMERS_IMPORT_ERROR
            }
        }

    def get_service_mode(self):
        """
        Determine the current operational mode of the service.

        Returns:
            str: One of the following modes based on component availability:
                - 'production': All core components are available.
                - 'partial': Some components are available.
                - 'development': No components are available.
        """
        if (self.components['eeg_preprocessing']['available'] and
            self.components['eeg_model']['available']):
            return 'production'
        elif any(comp['available'] for comp in self.components.values()):
            return 'partial'
        else:
            return 'development'

    def get_errors(self):
        """
        Retrieve a dictionary of components with active errors.

        Returns:
            dict: A mapping of component names to their error messages or flags,
                  including only those components currently in an error state.
        """
        return {name: comp['error'] for name, comp
                in self.components.items() if comp['error']}

# Initialize status manager
status_manager = ComponentStatus()

# === Initialize FastAPI App ===
app = FastAPI(
    title="NeuroCheck Backend",
    description="Neurocheck Backend for EEG and Alzheimers",
    version="0.5.1"
)

# === CORS Functionality for Independent Front End, Back End Communication ===
# https://fastapi.tiangolo.com/tutorial/cors/#use-corsmiddleware

# Define Sources Assigned Full BackEnd API Calling Capabilities
# TODO: Revisit removing localhost capabilities after successful debugged deployment
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
        - status: Service availability ("online", "degraded")
        - version: Current API version
        - message: User-friendly status message
        - mode: Operational mode ("production", "partial", "development")
        - components: Component availability status for each service
        - endpoints: Available API endpoints for discovery
    """
    mode = status_manager.get_service_mode()
    service_status = "online" if mode == "production" else "degraded"

    return {
        "status": service_status,
        "version": "0.5.0",
        "message": "Backend is running!",
        "mode": mode,
        "components": {name: comp['available'] for name, comp in status_manager.components.items()},
        "endpoints": ["/health", "/debug", "/predict/eeg", "/predict/alzheimers"]
    }

# === Debug Endpoint ===
@app.get("/debug")
def debug_status():
    """
    Returns comprehensive system status information for debugging.

    Provides detailed component status, operational mode, and specific error
    messages for troubleshooting component failures. Use this endpoint when
    components fail to load or when the service is in degraded mode.

    Returns:
        dict: Debug information containing:
        - mode: Current operational mode
        - components: Detailed component status and error information
        - errors: Active error messages or success confirmation
    """
    errors = status_manager.get_errors()
    return {
        "mode": status_manager.get_service_mode(),
        "components": status_manager.components,
        "errors": errors if errors else "No errors: All components loaded successfully"
    }

# === EEG Prediction Endpoint ===
@app.post("/predict/eeg")
async def predict_eeg(eeg_file: UploadFile = File(...)):
    """
    EEG prediction endpoint with comprehensive error handling and fallback responses.

    Accepts:
        - CSV file uploads

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
            eeg_contents = await eeg_file.read()
            eeg_df = pd.read_csv(BytesIO(eeg_contents))
            eeg_df.columns = eeg_df.columns.str.strip()
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
    preprocessing_eeg_success = False
    proc_eeg_df = eeg_df

    if PREPROCESS_EEG_AVAILABLE and PREPROCESS_EEG:
        try:
            proc_eeg_df = PREPROCESS_EEG(eeg_df)
            logging.info("Preprocessing completed successfully")
            preprocessing_eeg_success = True
        except (ValueError, RuntimeError) as e:
            logging.warning("Preprocessing failed: %s", e)

    # Step 3: Try predicting
    if MODEL_EEG_LOADED and MODEL_EEG_RUN:
        try:
            proc_eeg_df.columns = proc_eeg_df.columns.str.strip()
            logging.info("Using columns for prediction: %s", proc_eeg_df.columns.tolist())
            proba_eeg = float(MODEL_EEG_RUN.predict_proba(proc_eeg_df)[0, 1])
            prediction_eeg = int(proba_eeg > 0.45)  # apply custom threshold 0.45 probability >0.45 you get fatigued below - 0 not fatigued.
            result = {
                "backend_status": "Production",
                "fatigue_class": str(prediction_eeg),
                "confidence": round(proba_eeg if prediction_eeg == 1 else 1 - proba_eeg, 4),
                "filename": user_eeg_file_name,
                "preprocessing_used": preprocessing_eeg_success
            }

        except (ValueError, RuntimeError, KeyError) as e:
            logging.warning("Prediction failed: %s", e)
            result = create_eeg_dummy_response(user_eeg_file_name, f"prediction_error: {str(e)}")

    else:
        result = create_eeg_dummy_response(user_eeg_file_name, "Development")

    return result

# === Local Dev Server Runner ===
if __name__ == "__main__":
    uvicorn.run("neurocheck.api_folder.api_file:app", host="0.0.0.0", port=8000, reload=True)
