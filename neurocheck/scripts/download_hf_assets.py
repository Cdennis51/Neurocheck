import os
import shutil
from transformers import pipeline, AutoImageProcessor

# Config
MODEL_ID = "DHEIVER/Alzheimer-MRI"
CACHE_DIR = "./neurocheck/hf_cache"

def download_model():
    os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
    print(f"Using cache directory: {CACHE_DIR}")

    # Download processor
    print("Downloading processor...")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)

    # Download model via pipeline
    print("Downloading model and pipeline...")
    classifier = pipeline("image-classification", model=MODEL_ID, feature_extractor=processor)

    print("Download complete.")

def clear_cache():
    if os.path.exists(CACHE_DIR):
        print(f"Deleting cache directory: {CACHE_DIR}")
        shutil.rmtree(CACHE_DIR)
    else:
        print("Cache directory does not exist. Nothing to delete.")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "clear_cache":
        clear_cache()
    else:
        download_model()
