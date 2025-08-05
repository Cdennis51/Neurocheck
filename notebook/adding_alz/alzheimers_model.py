import base64
from io import BytesIO
import os
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification, pipeline
from pytorch_grad_cam.utils.image import show_cam_on_image

# === Set Cache Directory Before Model Loading ===
CACHE_DIR = "./neurocheck/hf_cache"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

# === Device Selection ===
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#TODO: return to mps
device = torch.device("cpu")  # Temporarily disable MPS

# === Model Loading Function ===
def load_model():
    """Load the model and processor, allowing for retries and error handling"""
    model = ViTForImageClassification.from_pretrained(
        "DHEIVER/Alzheimer-MRI",
        cache_dir=CACHE_DIR,
        local_files_only=False  # Allow downloads
    )
    processor = ViTImageProcessor.from_pretrained(
        "DHEIVER/Alzheimer-MRI",
        cache_dir=CACHE_DIR,
        local_files_only=False
    )
    return {"model": model, "processor": processor}

# === Load Models ===
try:
    model_dict = load_model()
    model = model_dict["model"].to(device)
    processor = model_dict["processor"]

    # Create classifier pipeline
    classifier = pipeline(
        "image-classification",
        model=model,
        feature_extractor=processor
    )
except Exception as e:
    raise RuntimeError(f"Failed to load model/processor: {e}")

# === Utility Functions ===
def create_brain_mask(image: Image.Image):
    img_array = np.array(image.convert("L"))
    return img_array > 30

def generate_attention_overlay(resized_image: Image.Image, original_image: Image.Image) -> np.ndarray:
    """
    Generate attention overlay from resized image and paste it into original image.

    Args:
        resized_image (PIL.Image): 224x224 image used for model attention
        original_image (PIL.Image): Full-size image for final overlay composition

    Returns:
        np.ndarray: Overlay image (same size as original_image)
    """
    brain_mask = create_brain_mask(resized_image)

    inputs = processor(images=resized_image.convert("RGB"), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = torch.stack(outputs.attentions)
    avg_attn = attentions[-1][0].mean(0)
    cls_map = avg_attn[0, 1:].reshape(14, 14).cpu().numpy()

    attn_resized = cv2.resize(cls_map, resized_image.size)
    attn_masked = attn_resized * brain_mask.astype(float)

    if attn_masked.max() > attn_masked.min():
        attn_normalized = (attn_masked - attn_masked.min()) / (attn_masked.max() - attn_masked.min())
    else:
        attn_normalized = attn_masked

    # Create overlay on resized input
    resized_np = np.array(resized_image.convert("RGB")).astype(np.float32) / 255.0
    overlay_resized = show_cam_on_image(resized_np, attn_normalized, use_rgb=True)

    # === Resize overlay to original image size ===
    overlay_upscaled = Image.fromarray(overlay_resized).resize(original_image.size)
    overlay_final = np.array(overlay_upscaled)

    return overlay_final

# === Predict Function ===
def predict(resized_image: Image.Image, original_image: Image.Image):
    """
    Run prediction on resized image, generate attention overlay on original image.

    Args:
        resized_image (PIL.Image): 224x224 preprocessed MRI image for model input
        original_image (PIL.Image): Original uploaded image (used for overlay)

    Returns:
        tuple: (result_dict, overlay_base64_str)
    """
    result = classifier(resized_image)[0]
    overlay = generate_attention_overlay(resized_image, original_image)

    overlay_pil = Image.fromarray(overlay.astype(np.uint8))
    buffer = BytesIO()
    overlay_pil.save(buffer, format="PNG")
    overlay_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return result, overlay_base64
