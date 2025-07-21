from io import BytesIO
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import uvicorn
import logging
import tempfile

# EDF file support requires mne package
# Attempt to import mne for EDF file support
# try:
#     import mne  # for EDF files
#     EDF_SUPPORT = True
# except ImportError:
#     EDF_SUPPORT = False

# Attempt to import preprocessing components
# If they fail, set to False and print a warning.
# This allows the backend to run even if preprocessing is not available.
PREPROCESS_AVAILABLE = None
try:
    from neurocheck.ml_logic.preprocess import preprocess_predict as preprocess_eeg
    PREPROCESS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Preprocessing not available. {e}")
    preprocess_eeg = None
    PREPROCESS_AVAILABLE = False

# Attempt to import model components that will be instantiated and called later.
# If they fail, set to False and print a warning.
# This allows the backend to run even if model is not available.
MODEL_AVAILABLE = None
try:
    from neurocheck.ml_logic.model import load_model as ml_load_eeg_model
    from neurocheck.ml_logic.model import predict_model as eeg_model_predict
    MODEL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Model components not available. {e}")
    ml_load_eeg_model = None
    eeg_model_predict = None
    MODEL_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="NeuroCheck Backend",
    description="EEG Fatigue Prediction Backend",
    version="0.2.0"
)

# Allow CORS so separate Streamlit frontend can call the backend
# https://fastapi.tiangolo.com/tutorial/cors/#use-corsmiddleware
app.add_middleware(
    CORSMiddleware,

    # For development, allow all origins.
    # For production, specify and updated allowed origins.
    allow_origins=["*"],
    # allow_origins=["https://your-frontend-url.com"],  # Replace with your frontend URL
    # allow_origins=["http://localhost:8501"],  # For local

    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Try to load model, set status flags
# If they fail, set to False and print a warning.
eeg_model = None
MODEL_LOADED = None
if MODEL_AVAILABLE:
    try:
        eeg_model = ml_load_eeg_model() # Ignore None Warning During Development
        MODEL_LOADED = True
        logging.info("Model loaded successfully")
        print("Model loaded successfully")
    except (FileNotFoundError, IOError) as e:
        print(f"Warning: Failed to load model. {e}")
        MODEL_LOADED = False


# Health check endpoint (Backend online checkpoint)
@app.get("/")
def index():
    """
    Health check endpoint for the NeuroCheck backend.

    Returns:
        dict: JSON response confirming backend is online,
              includes backend version and available prediction endpoint,
              as well as preprocessing and model status.
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

# EEG prediction endpoint (placeholder for now)
@app.post("/predict/eeg")
async def predict_eeg(file: UploadFile = File(...)):
    """
    EEG prediction with fallback to dummy responses.

    Accepts:
        - CSV, EDF file upload (EEG signal data)

    Process:
        - Reads uploaded CSV → converts to DataFrame
        - (Future) Preprocesses data before prediction
        - (Future) Uses ML model to predict fatigue level

    Returns:
        dict: JSON with dummy fatigue_class + confidence
    """

    filename = file.filename.lower()
    try:
        if filename.endswith(".csv"):
            eeg_df = pd.read_csv(file.file)
        elif filename.endswith(".edf"):
            eeg_df = read_edf_to_dataframe(file.file)
        else:
            return {"error": "Unsupported file format. Please upload CSV or EDF."}
    except Exception as e:
        return {"error": f"Failed to read EEG file: {str(e)}"}

    #  Read uploaded file as bytes → convert to DataFrame
    contents = await file.read()
    eeg_df = pd.read_csv(BytesIO(contents))

    # Try to preprocess the EEG data
    if PREPROCESS_AVAILABLE:
        try:
            proc_eeg_df = preprocess_eeg(eeg_df)
            preprocessing_success = True
        except (ValueError, KeyError, AttributeError, RuntimeError) as e:
            print(f"Preprocessing failed: {e}")
            #proc_eeg_df = eeg_df  # use raw data
            preprocessing_success = False
    else:
        proc_eeg_df = eeg_df
        preprocessing_success = False
    if MODEL_LOADED and eeg_model:
        try:
            prediction = eeg_model_predict(eeg_model, proc_eeg_df)
            # Convert numpy to native Python types for JSON
            result = {
                "backend_status": "production",
                "fatigue_class": str(prediction["class"]),
                "confidence": float(prediction["confidence"]),
                "filename": file.filename,
                "preprocessing_used": preprocessing_success
            }
        except (ValueError, KeyError, AttributeError, RuntimeError) as e:
            print(f"Prediction failed: {e}")
            result = create_dummy_response(file.filename or "unknown_file", "prediction_error")
    else:
        result = create_dummy_response(file.filename or "unknown_file", "development")

    return result

def create_dummy_response(filename: str, mode: str):
    """Generate dummy response for development/testing"""
    import random

    dummy_classes = ["alert", "fatigued", "drowsy"]

    return {
        "backend_status": f"development_mode_{mode}",
        "fatigue_class": random.choice(dummy_classes),
        "confidence": round(random.uniform(0.7, 0.95), 2),
        "filename": filename,
        "note": f"Dummy response - {mode}. Real model integration pending."
    }

# Local development server
if __name__ == "__main__":
    uvicorn.run("neurocheck.api_file:app", host="0.0.0.0", port=8000, reload=True)
