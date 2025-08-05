from transformers import pipeline, AutoImageProcessor
import torch

# Paths
cache_dir = "./neurocheck/hf_cache/models--DHEIVER--Alzheimer-MRI/snapshots/f947f2c031369d346e9c46435ec0e4b1c1936261"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load processor and pipeline from local snapshot folder
processor = AutoImageProcessor.from_pretrained(cache_dir)
classifier = pipeline("image-classification", model=cache_dir, feature_extractor=processor, device=device.index if device.type != "cpu" else -1)
model = classifier.model.to(device)

def predict(image):
    result = classifier(image)[0]
    return result

if __name__ == "__main__":
    from PIL import Image
    #TODO: Update path below and uncomment
    #img = Image.open("/Users/db/Desktop/image.png")  # update path as needed
    print(predict(img))
