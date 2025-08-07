from transformers import pipeline, AutoImageProcessor
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np
import torch
import cv2  # pylint: disable=no-member
import base64
from io import BytesIO
from torch.nn.functional import softmax

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Global variables for lazy loading
classifier = None
processor = None
model = None

def _load_model():
    """Load model components on first use"""
    global classifier, processor, model
    if classifier is None:
        classifier = pipeline("image-classification",
                             model="DHEIVER/Alzheimer-MRI",
                             device=device.index if device.type != "cpu" else -1)
        processor = AutoImageProcessor.from_pretrained("DHEIVER/Alzheimer-MRI")
        model = classifier.model.to(device)

def generate_attention_overlay(image: Image.Image) -> np.ndarray:
    _load_model()  # Ensure model is loaded
    # Prepare input
    inputs = processor(images=image.convert("RGB"), return_tensors="pt", do_resize=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass with attention
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Process attention map
    attentions = torch.stack(outputs.attentions)  # pylint: disable=no-member
    last_layer_attn = attentions[-1][0]                     # (heads, tokens, tokens)
    avg_attn = last_layer_attn.mean(0)                      # (tokens, tokens)
    cls_attn = avg_attn[0, 1:]                              # attention from CLS to all patches
    cls_map = cls_attn.reshape(14, 14).cpu().numpy()        # (14x14)

    # Resize and normalize
    attn_resized = cv2.resize(cls_map, (224, 224))  # pylint: disable=no-member
    attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min())

    # Prepare original image
    image_resized = image.convert("RGB").resize((224, 224))
    img_np = np.array(image_resized).astype(np.float32) / 255.0

    # Overlay
    overlay = show_cam_on_image(img_np, attn_resized, use_rgb=True)

    return overlay

def predict_alzheimers_image(resized_image):
    _load_model()  # Load on first use

    # Prepare inputs
    inputs = processor(images=resized_image.convert("RGB"), return_tensors="pt", do_resize=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1)[0]

    # Get top class
    pred_idx = probs.argmax().item()
    confidence = probs[pred_idx].item()
    label = model.config.id2label[pred_idx]

    # === Overlay generation ===
    overlay = generate_attention_overlay(resized_image)

    # Convert overlay to base64 PNG string
    overlay_pil = Image.fromarray(overlay.astype(np.uint8))
    buffer = BytesIO()
    overlay_pil.save(buffer, format="PNG")
    overlay_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {"label": label, "score": confidence}, overlay_base64
