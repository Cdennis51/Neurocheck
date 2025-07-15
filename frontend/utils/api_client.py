import requests
import streamlit as st

def call_eeg_api(uploaded_file):
    files = {"file": uploaded_file}
    response = requests.post("http://localhost:8000/predict/eeg", files=files)
    return response.json()

# In your app.py
if uploaded_file:
    result = call_eeg_api(uploaded_file)
    st.write(f"Prediction: {result['prediction']}")
