from fastapi import FastAPI

app = FastAPI()

# Root `/` endpoint
@app.get('/')
def index():
    return {'Welcome to the Neurocheck API!': True}

#Prediction endpoint
@app.get('/predict')
def predict():
    return {'This is the prediction endpoint': True}
