from transformers import AutoModelForImageClassification, AutoImageProcessor

model_id = "DHEIVER/Alzheimer-MRI"
cache_dir = "./neurocheck/hf_cache"  # relative to current working dir

# Download model and processor files to cache_dir
model = AutoModelForImageClassification.from_pretrained(model_id, cache_dir=cache_dir)
processor = AutoImageProcessor.from_pretrained(model_id, cache_dir=cache_dir)

print("Download complete.")
