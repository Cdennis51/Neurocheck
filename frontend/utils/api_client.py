import requests

# API_BASE_URL = "http://localhost:8000"

def call_eeg_api(uploaded_file):
    """Send EEG file to backend FastAPI for prediction."""
    # Convert user uploaded file into (filename, bytes, MIME type)
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type
        )
    }
    # Make the API call to the EEG prediction endpoint
    try:
        response = requests.post(f"{API_BASE_URL}/predict/eeg", files=files, timeout=30)
        response.raise_for_status() # Raise HTTP failures
        return response.json()      # Return JSON response from API
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}    # raises file processing errors
