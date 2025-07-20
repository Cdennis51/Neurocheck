"""Utility for sending EEG data to a FastAPI backend for fatigue prediction.
This module provides a utility function to send EEG data files to a FastAPI backend
for fatigue prediction. It attempts to contact the `/predict/eeg` endpoint and returns
the JSON response. If the backend is unavailable, it returns a mock response for demo purposes.
"""
import requests

API_BASE_URL = "http://localhost:8000" # Replace with deployed backend later

def call_eeg_api(uploaded_file):
    """Send EEG file to backend FastAPI for prediction.
       If backend is offline, return a mock response for demo."""

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

    ## Backend online error handling network issues or server errors
    #except requests.exceptions.RequestException as e:
    #    return {"error": str(e)}    # raises file processing errors

    # Current Development Handling of Backend offline → return fake response
    except requests.exceptions.RequestException:
        return {
            "fatigue_class": "fatigued (demo)",
            "confidence": 0.87,
            "backend_status": "offline"
        }
