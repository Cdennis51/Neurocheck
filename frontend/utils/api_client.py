"""Utility for sending EEG data to a FastAPI backend for fatigue prediction.
This module provides a utility function to send EEG data files to a FastAPI backend
for fatigue prediction. It attempts to contact the `/predict/eeg` endpoint and returns
the JSON response. If the backend is unavailable, it returns a mock response for demo purposes.
"""
import requests


# Default backend URL (can override with ENV variable)
import streamlit as st
BACKEND_API = st.secrets["GCP_URL"]

def check_backend_health():
    """Ping backend/health endpoint."""
    try:
        resp = requests.get(f"{BACKEND_API}/health", timeout=3)
        if resp.status_code == 200:
            return resp.json()
        else:
            return {"status": "offline"}
    except requests.exceptions.RequestException:
        return {"status": "offline with request exception"}


def call_eeg_api(uploaded_file, timeout: int = 120):
    """
        Send an EEG file to the FastAPI backend for fatigue prediction.

        Args:
            uploaded_file: File-like object (e.g., from Streamlit `st.file_uploader`)
                        Must have `.name`, `.getvalue()`, and `.type` attributes.
            timeout (int): Request timeout in seconds (default: 10).

        Returns:
            dict: JSON response from backend if available.
                Example:
                {
                    "backend_status": "production",
                    "fatigue_class": "alert",
                    "confidence": 0.92,
                    "filename": filename,
                    ...
                }

                If backend is offline/unreachable:
                {
                    "fatigue_class": "fatigued (demo)",
                    "confidence": 0.87,
                    "backend_status": "offline"
                }

        Raises:
            ValueError: If uploaded_file is not a valid file-like object.
        """

    # Validate uploaded file
    if not hasattr(uploaded_file, "getvalue"):
        raise ValueError("uploaded_file must be a file-like object (e.g., from Streamlit uploader)")

    # Convert file for multipart/form-data upload
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream"
        )
    }

    try:
        # Send to backend prediction endpoint
        headers = {
        "Authorization": f"Bearer {st.secrets['GCLOUD_ACCESS_TOKEN']}"
        }
        response = requests.post(f"{BACKEND_API}/predict/eeg", files=files, headers=headers, timeout=timeout)
        response.raise_for_status() # Raise HTTP failures
        return response.json()      # Return JSON response from API

    except requests.exceptions.HTTPError as e:
        # Backend responded with error (e.g., 400 Bad Request)
        return {"backend_status": "error", "message": f"Backend error: {str(e)}"}

    except requests.exceptions.RequestException:
        # Backend unreachable → return dummy response
        return {
            "fatigue_class": "Fatigued (We're having issues connecting to our server at the moment)",
            "confidence": 0.87,
            "backend_status": "offline"
        }
