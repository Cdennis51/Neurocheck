"""Frontend Streamlit app for EEG Fatigue Detector.
Accepts user uploads in CSV and EDF formats and displays predictions.
"""
import streamlit as st
from utils.api_client import call_eeg_api

# Set up the Streamlit app
st.set_page_config(page_title="EEG Fatigue Detector", layout="centered")
st.title("NeuroCheck EEG Fatigue Detector")

# Display instructions to user
uploaded_file = st.file_uploader("📂 Upload your EEG File Here", type=["csv"])

if uploaded_file:
    # Display user upload status
    st.write(f"✅ File uploaded: {uploaded_file.name}")

    # Display file processing status
    with st.spinner("Analyzing EEG data..."):
        result = call_eeg_api(uploaded_file)

    # If backend offline, warn but still provide user with fake response
    if result.get("backend_status") == "offline":
        st.warning("⚠️ Backend is offline, showing demo prediction instead.")

    # Display prediction result
    if "fatigue_class" in result:
        st.success(f"Prediction: **{result['fatigue_class']}**")
        if "confidence" in result:
            st.write(f"Confidence: {result['confidence']:.2f}")
    else:
        st.error("❌ Could not get prediction.")
