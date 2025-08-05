
import base64
from io import BytesIO
import os
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification, pipeline
from pytorch_grad_cam.utils.image import show_cam_on_image

# === Cache Directory ===
CACHE_DIR = "./neurocheck/hf_cache"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

# === Device Selection ===
device = torch.device("cpu")  # Change to MPS if ready

# === Load Model ===
model_name = "DHEIVER/Alzheimer-MRI"
model = ViTForImageClassification.from_pretrained(model_name, cache_dir=CACHE_DIR, local_files_only=False)
processor = ViTImageProcessor.from_pretrained(model_name, cache_dir=CACHE_DIR, local_files_only=False)
classifier = pipeline("image-classification", model=model, feature_extractor=processor)
model = classifier.model.to(device)

# === Helpers ===
def create_brain_mask(image: Image.Image):
    gray = np.array(image.convert("L"))
    return gray > 30

def generate_attention_overlay(resized: Image.Image, original: Image.Image):
    brain_mask = create_brain_mask(resized)
    inputs = processor(images=resized.convert("RGB"), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = torch.stack(outputs.attentions)
    avg_attn = attentions[-1][0].mean(0)
    cls_map = avg_attn[0, 1:].reshape(14, 14).cpu().numpy()
    attn_resized = cv2.resize(cls_map, resized.size)
    attn_masked = attn_resized * brain_mask.astype(float)

    if attn_masked.max() > attn_masked.min():
        attn_normalized = (attn_masked - attn_masked.min()) / (attn_masked.max() - attn_masked.min())
    else:
        attn_normalized = attn_masked

    input_np = np.array(resized.convert("RGB")).astype(np.float32) / 255.0
    overlay_resized = show_cam_on_image(input_np, attn_normalized, use_rgb=True)
    overlay_upscaled = Image.fromarray(overlay_resized).resize(original.size)
    overlay_final = np.array(overlay_upscaled)

    return overlay_final

# === Main Predict Function ===
def predict_alzheimers_image(resized_image, original_image):
    inputs = processor(images=resized_image.convert("RGB"), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    probs = outputs.logits.softmax(dim=-1)
    label_id = probs.argmax().item()
    confidence = probs.max().item()
    label = model.config.id2label[label_id]

    overlay_final = generate_attention_overlay(resized_image, original_image)
    overlay_pil = Image.fromarray(overlay_final.astype(np.uint8))
    buffer = BytesIO()
    overlay_pil.save(buffer, format="PNG")
    overlay_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {"label": label, "score": confidence}, overlay_base64
