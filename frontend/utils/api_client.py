"""Utility for sending EEG and MRI data to a FastAPI backend for prediction.
This module provides a utility function to send EEG and MRI data files to a FastAPI backend.
This includes functions to:
- Check backend health
- Send EEG CSV files for fatigue prediction
- Send MRI image files files for Alzheimer classification.

All requests include authorization headers from Streamlit secrets.
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


def call_eeg_api(uploaded_eeg_file, timeout: int = 120):
    """
        Send an EEG file to the FastAPI backend for fatigue prediction.

        Args:
            uploaded_eeg_file: File-like object (from Streamlit uploader) with .name, .getvalue(), .type
            timeout (int): Request timeout in seconds (default: 120)

        Returns:
            dict: Backend JSON response or dummy offline response.
        """

    # Validate uploaded file
    if not hasattr(uploaded_eeg_file, "getvalue"):
        raise ValueError("Uploaded EEG file must be a file-like object (e.g., from Streamlit uploader)")

    # Convert file for multipart/form-data upload
    files = {
        "eeg_file": (
            uploaded_eeg_file.name,
            uploaded_eeg_file.getvalue(),
            uploaded_eeg_file.type or "application/octet-stream"
        )
    }

    headers = {
        "Authorization": f"Bearer {st.secrets['GCLOUD_ACCESS_TOKEN']}"
    }

    try:
        response = requests.post(
            f"{BACKEND_API}/predict/eeg",
            files=files,
            headers=headers,
            timeout=timeout
            )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        return {
            "fatigue_class": "Fatigued",
            "confidence": 0.87,
            "backend_status": "error" if isinstance(e, requests.exceptions.HTTPError)
                else "offline",
            "message": f"Backend error: {str(e)}"
                if isinstance(e, requests.exceptions.HTTPError) else None
        }

def call_mri_api(uploaded_image_file, timeout: int = 120):
    """
    Send an MRI image file to the FastAPI backend for Alzheimer classification.

    Args:
        uploaded_image_file: File-like object (from Streamlit uploader) with .name, .type, and readable content
        timeout (int): Request timeout in seconds (default: 120)

    Returns:
        dict: Backend JSON response or error dict.
    """
    if not hasattr(uploaded_image_file, "read") and not hasattr(uploaded_image_file, "getvalue"):
        raise ValueError("Image filefile must be a file-like object (e.g., from Streamlit uploader)")

    # Use getvalue() if available, else fallback to read()
    file_content = uploaded_image_file.getvalue() if hasattr(uploaded_image_file, "getvalue") else uploaded_image_file.read()

    files = {
        "file": (
            uploaded_image_file.name,
            file_content,
            uploaded_image_file.type or "application/octet-stream"
        )
    }

    headers = {
        "Authorization": f"Bearer {st.secrets['GCLOUD_ACCESS_TOKEN']}"
    }

    try:
        response = requests.post(
            f"{BACKEND_API}/predict/alzheimers", # want to add correct URL
            files=files,
            headers=headers,
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.HTTPError as e:
        return {"error": f"Backend error: {str(e)}"}

    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
