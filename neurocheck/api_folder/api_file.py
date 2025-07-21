from fastapi import FastAPI, File, UploadFile
import pandas as pd
from io import BytesIO
from neurocheck.ml_logic.preprocess import preprocess_predict

app = FastAPI()

# Root `/` endpoint
@app.get('/')
def index():
    return {'Welcome to the Neurocheck API!': True}

#Prediction endpoint
@app.post('/predict/eeg')
async def predict(file: UploadFile = File(...)):
    # Read uploaded file as bytes and convert to DataFrame
    contents = await file.read()
    df = pd.read_csv(BytesIO(contents))

    # Apply preprocessing if needed
    df = preprocess_predict(df)

    # Make predictions
    #prediction = model.predict(df)

    #return {'prediction': prediction.tolist()}  # Convert numpy to list for JSON response

    return {'This is the prediction endpoint': True}
