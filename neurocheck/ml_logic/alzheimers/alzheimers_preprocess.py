import io
from PIL import Image


def resize_upload(uploaded_image_bytes):
    """
    Prepare both original and resized versions of the uploaded image.

    Args:
        uploaded_image_bytes (bytes): Raw image file content from UploadFile.read()

    Returns:
        tuple:
            - original_image (PIL.Image): Original grayscale image
            - resized_image (PIL.Image): Resized image (224x224) for model input
    """
    image = Image.open(io.BytesIO(uploaded_image_bytes)).convert('RGB')
    original_image = image.copy()
    resized_image = image.resize((224, 224))  # Model expects 224x224

    return original_image, resized_image
