import streamlit as st
from utils.api_client import call_eeg_api

st.title("NeuroCheck EEG Fatigue Detector MVP")

uploaded_file = st.file_uploader("Upload EEG data", type=["csv", "edf"])

if uploaded_file:
    # Display upload status
    st.write(f"✅ File uploaded: {uploaded_file.name}")
    # Display processing status
    with st.spinner("Analyzing EEG data..."):
        result = call_eeg_api(uploaded_file)
    # API error handling
    if "error" in result:
        st.error(f"API Error: {result['error']}")
    else:
        # Display prediction with confidence levels
        st.success(f"Prediction: **{result['fatigue_class']}**")
        st.write(f"Confidence: {result['confidence']:.2f}")
