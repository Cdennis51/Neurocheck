from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


app = FastAPI(
    title="NeuroCheck Backend",
    description="EEG Fatigue Prediction Backend",
    version="0.1.0"
)

# Allow CORS so Streamlit frontend can talk to backend
#https://fastapi.tiangolo.com/tutorial/cors/
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For MVP, allow all. Later restrict to frontend domain.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    """Returns a simple confirmation that the backend is running."""
    return {"message": "Backend is running! Use /predict/eeg for predictions."}

@app.post("/predict/eeg")
async def predict_eeg(file: UploadFile = File(...)):
    """
    Dummy EEG prediction endpoint.
    - Accepts an uploaded file.
    - Currently returns a placeholder response.
    """

    # Preprocessing and model inference will later be implemented here.
    filename = file.filename

    #file_content = await file.read()  # returns raw bytes if needed for debugging

    #This is a placeholder response. Later, this will be replaced with actual model predictions.
    result = {
        "backend_status": "online",
        "fatigue_class": "fatigued",    # rename to "fatigue_class" for consistency with frontend
        "confidence": 0.87,
        "filename": filename,            # optional
        "note": "You are receiving a test response placeholder."
    }
    return result

if __name__ == "__main__":
    # Local dev server
    # Run the FastAPI app with Uvicorn when this file is executed directly.
    # "api.main:app" points to the FastAPI instance named 'app' inside api/main.py.
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
