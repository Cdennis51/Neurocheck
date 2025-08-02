from PIL import Image

def resize_upload(uploaded_image_bytes):

    #Convert to PIL image
    image = Image.open(io.BytesIO(uploaded_image_bytes))
    image = image.convert('L')
    image = image.resize((224,224)) # Must be a tuple.

    return image


# Method 1: FastAPI with file upload
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

app = FastAPI()

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):

    # Read the uploaded file
    contents = await file.read()

    image = resize_upload(contents)

    print(f"Processed image: {image}")  # Will show: <PIL.Image.Image image mode=L size=128x128>

    # Use with classifier
    result = predict(image)

    return {"Prediction": result}
