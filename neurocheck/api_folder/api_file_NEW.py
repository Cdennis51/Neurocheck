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
# === EDF Reader Function (Post MVP) ===
def read_edf_to_dataframe(file_obj):
    raise NotImplementedError("EDF file support not yet implemented. Use CSV instead.")
# === Import Preprocessing ===
try:
    from neurocheck.ml_logic.preprocess import preprocess_eeg_df as preprocess_eeg
    PREPROCESS_AVAILABLE = True
except ImportError as e:
    logging.warning("Preprocessing not available: %s", e)
    preprocess_eeg = None
    PREPROCESS_AVAILABLE = False
# === Import Model ===
try:
    from neurocheck.ml_logic.registry import retrieve_model as ml_load_eeg_model
    MODEL_AVAILABLE = True
except ImportError as e:
    logging.warning("Model components not available: %s", e)
    ml_load_eeg_model = None
    MODEL_AVAILABLE = False
# === Load Model At Startup ===
MODEL_LOADED = False
eeg_model = None
if MODEL_AVAILABLE:
    try:
        eeg_model = ml_load_eeg_model()
        if eeg_model is None:
            logging.warning("Model loaded but is None.")
        else:
            MODEL_LOADED = True
            logging.info("Model loaded successfully")
    except (FileNotFoundError, IOError, ValueError, RuntimeError) as e:
        logging.warning("Failed to load model: %s", e)
# === Initialize FastAPI App ===
app = FastAPI(
    title="NeuroCheck Backend",
    description="EEG Fatigue Prediction Backend",
    version="0.4.0"
)
# === CORS Settings ===
ALLOWED_ORIGINS = [
    "http://localhost:8501",
    "https://neurocheck-frontend.streamlit.app/",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# === Dummy Response Generator ===
def create_dummy_response(filename: str, mode: str):
    dummy_classes = ["fatigued", "not fatigued"]
    return {
        "backend_status": f"development_mode_{mode}",
        "fatigue_class": random.choice(dummy_classes),
        "confidence": round(random.uniform(0.7, 0.95), 2),
        "filename": filename,
        "note": f"Dummy response - {mode}. Real model integration pending."
    }
# === Health Check ===
@app.get("/health")
def index():
    return {
        "status": "online",
        "version": "0.1.0",
        "message": "Backend is running! Use /predict/eeg for predictions.",
        "components": {
            "preprocessing": PREPROCESS_AVAILABLE,
            "model": MODEL_LOADED,
            "mode": "production" if (PREPROCESS_AVAILABLE and MODEL_LOADED) else "development"
        }
    }
# === Prediction Endpoint ===
@app.post("/predict/eeg")
async def predict_eeg(file: UploadFile = File(...)):
    filename = file.filename.lower()
    try:
        if filename.endswith(".csv"):
            contents = await file.read()
            eeg_df = pd.read_csv(BytesIO(contents))
            eeg_df.columns = eeg_df.columns.str.strip()
        elif filename.endswith(".edf"):
            eeg_df = read_edf_to_dataframe(file.file)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload CSV (EDF coming soon)."
            )
    except Exception as e:
        logging.error("Failed to read EEG file: %s", e)
        raise HTTPException(status_code=400, detail=f"EEG file read error: {e}")
    # === Preprocess EEG ===
    proc_eeg_df = eeg_df
    preprocessing_success = False
    if PREPROCESS_AVAILABLE and preprocess_eeg:
        try:
            proc_eeg_df = preprocess_eeg(eeg_df)
            preprocessing_success = True
        except Exception as e:
            logging.warning("Preprocessing failed, using raw EEG: %s", e)
    # === Predict ===
    if MODEL_LOADED and eeg_model:
        try:
            proc_eeg_df.columns = proc_eeg_df.columns.str.strip()
            logging.info(f"Using columns for prediction: {proc_eeg_df.columns.tolist()}")
            prediction = eeg_model.predict(proc_eeg_df)
            probability = float(eeg_model.predict_proba(proc_eeg_df)[0, 1])
            return {
                "backend_status": "production",
                "fatigue_class": str(prediction[0]),
                "confidence": probability,
                "filename": file.filename,
                "preprocessing_used": preprocessing_success
            }
        except Exception as e:
            logging.warning("Prediction failed: %s", e)
            return create_dummy_response(file.filename, "prediction_error")
    else:
        return create_dummy_response(file.filename, "development")
# === Run Dev Server ===
if __name__ == "__main__":
    uvicorn.run("neurocheck.api_folder.api_file_NEW:app", host="0.0.0.0", port=8000, reload=True)
