"""Tabbed Streamlit frontend for NeuroCheck:
- EEG tab accepts CSV uploads for fatigue detection.
- MRI tab accepts JPG, JPEG, or PNG uploads for Alzheimer classification.
After upload, each module sends data to a shared backend and displays model predictions.
"""
import streamlit as st
from utils.api_client import call_eeg_api
from PIL import Image
# import io  # Uncomment later if sending image bytes to backend

st.set_page_config(page_title="NeuroCheck", layout="centered")

#layyout, colours, fonts
def inject_css():
    st.markdown("""
        <style>
        body {
            background-color: #f7f9fc;
            font-family: 'Segoe UI', sans-serif;
        }
        header, footer {visibility: hidden;}
        .block-container {
            padding-top: 2rem;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #eaf4f4;
            border-radius: 6px;
            padding: 10px;
            margin-right: 5px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #205375 !important;
            color: white !important;
        }
        div.stButton > button {
            background-color: #205375;
            color: white;
            border-radius: 8px;
            font-weight: bold;
            padding: 0.5em 1em;
        }
        .stFileUploader label {
            font-weight: 600;
            color: #205375;
        }
        .result-card {
            background-color: #eaf4f4;
            padding: 1rem;
            border-left: 5px solid #205375;
            border-radius: 10px;
            margin-top: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)


inject_css()  # Call it at the top of main script
# Set up the Streamlit app with two tabs

tab1, tab2 = st.tabs(["EEG Fatigue Detector", "Alzheimer MRI Classifier"])

# === EEG Tab ===
with tab1:
    st.subheader("EEG Fatigue Detector")

    # Display EEG upload instructions to user
    uploaded_eeg_file = st.file_uploader("📂 Upload an EEG", type=["csv"])

    if uploaded_eeg_file:
        # Display user EEG file upload status
        st.write(f"✅ File uploaded: {uploaded_eeg_file.name}")

        # Display EEG file processing status
        with st.spinner("Analyzing EEG data..."):
            result = call_eeg_api(uploaded_eeg_file)

        # If backend offline, warn but still provide user with fake response
        if result.get("backend_status") == "offline":
            st.warning("⚠️ Backend is offline, showing demo prediction instead.")

        # Display prediction result
        if "fatigue_class" in result:
            # Convert numeric string to readable label
            fatigue_labels = {"0": "Not Fatigued", "1": "Fatigued"}
            display_result = fatigue_labels.get(result['fatigue_class'], result['fatigue_class'])

            #wraps prediction in styled card
            st.markdown(f"""
                <div class='result-card'>
                    <h3>Prediction: {display_result}</h3>
                    <p>Confidence Level: {result['confidence']:.2f}</p>
                </div>
            """, unsafe_allow_html=True)

            st.success(f"Prediction: **{display_result}**")
            if "confidence" in result:
                st.write(f"Confidence Level: {result['confidence']:.2f}")
        else:
            st.error("❌ Could not get prediction.")

# === MRI Tab ===
with tab2:
    st.subheader("Alzheimer MRI Classifier")
    # st.markdown("Upload an MRI brain image or use our sample to get a classification.")

    # Display EEG upload instructions to user
    uploaded_mri_file = st.file_uploader("📂 Upload an MRI Image", type=["jpg", "jpeg", "png"])

    # # Button to load sample image
    # use_sample = st.button("Use Sample Image")

    mri_image_to_classify = None

    if uploaded_mri_file:
        # Display user MRI file upload status
        mri_image_to_classify = Image.open(uploaded_mri_file)
        st.success(f"✅ File uploaded: {uploaded_mri_file.name}")
    # elif use_sample:
    #     sample_path = "sample_mri.jpg"  # You will add this file locally or via URL
    #     mri_image_to_classify = Image.open(sample_path)
    #     st.info("📁 Using sample image.")

    if mri_image_to_classify:
        # Display the uploaded or sample MRI image to the user
        st.image(mri_image_to_classify, caption="MRI Input", use_column_width=True)
    # TODO: Integrate backend call to Hugging Face model via FastAPI
    # This block will handle:
    #   - Sending the image to the /predict/alz endpoint
    #   - Receiving the classification result
    #   - Displaying the predicted label and confidence score

        # This additional section will upload and show the MRI Image, discplay prediction and confidence and display attention map.
        with st.spinner("Analyzing MRI..."):
            result = call_mri_api(uploaded_mri_file)

        if "error" in result:
            st.error(f"❌ Error: {result['error']}")
        else:
            st.markdown(f"""
                <div class='result-card'>
                    <h3>Prediction: {result['prediction']}</h3>
                    <p>Confidence: {result['confidence']:.2f}</p>
                </div>
            """, unsafe_allow_html=True)

            if "overlay" in result:
                st.image(
                    f"data:image/png;base64,{result['overlay']}",
                    caption="Attention Map Overlay",
                    use_column_width=True
                )
