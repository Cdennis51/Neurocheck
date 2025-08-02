# from transformers import pipeline

# def predict(preprocessed_image):

    # Create a pipeline
    # classifier = pipeline("image-classification", model="DHEIVER/Alzheimer-MRI")

    # Get label mappings from the model
    # try:
        # id2label = model.config.id2label
        # label2id = model.config.label2id

        # print(f"Number of classes: {len(id2label)}")
        # print(f"Classes: {id2label}")

    # except:
        # print("Failed to get label mappings from the model.")

    # Predict the label
    # result = classifier(preprocessed_image)

    # return result[0]


# Load once at module level
# classifier = pipeline("image-classification", model="DHEIVER/Alzheimer-MRI")

# def predict(preprocessed_image):
#     # Access model config
#     try:
#         model = classifier.model
#         id2label = model.config.id2label
#         label2id = model.config.label2id

#         print(f"Number of classes: {len(id2label)}")
#         print(f"Classes: {id2label}")
#     except Exception as e:
#         print(f"Failed to get label mappings: {e}")

#     # Predict
#     result = classifier(preprocessed_image)
#     return result[0]


from transformers import pipeline, AutoImageProcessor
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np
import torch
import cv2
import base64
from io import BytesIO

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load classifier and processor once
classifier = pipeline("image-classification", model="DHEIVER/Alzheimer-MRI")
processor = AutoImageProcessor.from_pretrained("DHEIVER/Alzheimer-MRI")
model = classifier.model.to(device)

def generate_attention_overlay(image: Image.Image) -> np.ndarray:
    # Prepare input
    inputs = processor(images=image.convert("RGB"), return_tensors="pt", do_resize=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass with attention
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Process attention map
    attentions = torch.stack(outputs.attentions)            # (layers, batch, heads, tokens, tokens)
    last_layer_attn = attentions[-1][0]                     # (heads, tokens, tokens)
    avg_attn = last_layer_attn.mean(0)                      # (tokens, tokens)
    cls_attn = avg_attn[0, 1:]                              # attention from CLS to all patches
    cls_map = cls_attn.reshape(14, 14).cpu().numpy()        # (14x14)

    # Resize and normalize
    attn_resized = cv2.resize(cls_map, (224, 224))
    attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min())

    # Prepare original image
    image_resized = image.convert("RGB").resize((224, 224))
    img_np = np.array(image_resized).astype(np.float32) / 255.0

    # Overlay
    overlay = show_cam_on_image(img_np, attn_resized, use_rgb=True)

    return overlay

def predict(preprocessed_image: Image.Image):
    result = classifier(preprocessed_image)[0]
    overlay = generate_attention_overlay(preprocessed_image)

    # Convert overlay to base64 PNG string
    overlay_pil = Image.fromarray(overlay.astype(np.uint8))
    buffer = BytesIO()
    overlay_pil.save(buffer, format="PNG")
    overlay_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return result, overlay_base64
