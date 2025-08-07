"""Tabbed Streamlit frontend for NeuroCheck:
- EEG tab accepts CSV uploads for fatigue detection.
- MRI tab accepts JPG, JPEG, or PNG uploads for Alzheimer classification.
After upload, each module sends data to a shared backend and displays model predictions.
"""
import streamlit as st
from utils.api_client import call_eeg_api, call_mri_api
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import io

#streamlit set up
st.set_page_config(page_title="NeuroCheck", layout="centered", page_icon="🧠")

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

        .result-card h3 {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }

        .result-card p {
            font-size: 1.2rem;
        }

        .navbar {
            background-color: #0b2545;
            padding: 1rem 2rem;
            color: white;
            font-size: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .navbar a {
            color: white;
            margin-left: 20px;
            text-decoration: none;
            font-weight: 500;
        }
        </style>
    """, unsafe_allow_html=True)


inject_css()  # Call it at the top of main script
# Set up the Streamlit app with two tabs
def render_navbar():
    st.markdown("""
        <div style="background-color:#0B2545; padding: 1rem 2rem; display: flex; justify-content: space-between; align-items: center; color: white; border-radius: 8px;">
            <div style="font-size: 1.5rem; font-weight: bold;">🧠 NeuroCheck</div>
            <div style="font-size: 1rem;">
                <a href="#" style="margin-right: 1.5rem; color: white; text-decoration: none;">EEG</a>
                <a href="#" style="margin-right: 1.5rem; color: white; text-decoration: none;">Alzheimer's</a>
                <a href="#" style="color: white; text-decoration: none;">Voice</a>
            </div>
        </div>
        <br>
    """, unsafe_allow_html=True)

render_navbar()
st.markdown("""
    <div style='background-color: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.05);'>
""", unsafe_allow_html=True)

st.markdown("## AI-Powered Neurological Screening")
st.markdown("Upload your neurological data and get rapid assessment results.")

#tabs
tab1, tab2 = st.tabs(["EEG Fatigue Detector", "Alzheimer MRI Classifier"])

# === EEG Tab ===
with tab1:
    st.subheader("EEG Fatigue Detector")

    # Initialize result variable
    result = None
    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown("""
            <div style='background-color: #f0f4f8; border: 2px dashed #205375; border-radius: 10px; padding: 1.5rem; text-align: center;'>
                <h4 style='color:#205375; margin-bottom: 1rem;'>📂 Upload your EEG</h4>
        """, unsafe_allow_html=True)

        # Place uploader *inside* styled div with label hidden
        uploaded_eeg_file = st.file_uploader(label="", type=["csv"], label_visibility="hidden")

        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded_eeg_file:
            st.success(f"✅ File uploaded: {uploaded_eeg_file.name}")

            with st.spinner("Analyzing EEG data..."):
                result = call_eeg_api(uploaded_eeg_file)

            if result.get("backend_status") == "offline":
                st.warning("⚠️ Backend is offline, showing demo prediction instead.")

            if "fatigue_class" in result:
                fatigue_labels = {"0": "Not Fatigued", "1": "Fatigued"}
                display_result = fatigue_labels.get(result['fatigue_class'], result['fatigue_class'])


            else:
                st.error("❌ Could not get prediction.")

        with col2:
            # Display EEG Signal using matplotlib (dynamic waveform)
            if uploaded_eeg_file:
                try:
                    df = pd.read_csv(uploaded_eeg_file)
                    time_col = df.columns[0]
                    signal_cols = df.columns[1:6]

                    fig, ax = plt.subplots(figsize=(6, 4))
                    for col in signal_cols:
                        ax.plot(df[time_col], df[col], label=col)
                    ax.set_title("EEG Signal")
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Amplitude")
                    ax.legend(loc="upper right")
                    st.pyplot(fig)

                    # 🧠 Output result box after plot
                    if result and "fatigue_class" in result:
                        fatigue_labels = {"0": "Not Fatigued", "1": "Fatigued"}
                        display_result = fatigue_labels.get(result['fatigue_class'], result['fatigue_class'])

                        st.markdown(f"""
                            <div class='result-card' style='text-align:center; margin-top: 1rem;'>
                                <div style='font-size: 1.2rem; font-weight: 600;'>Fatigue Score</div>
                                <div style='font-size: 3rem; font-weight: bold; color: #205375;'>{result['confidence']:.2f}</div>
                                <div style='font-size: 1.2rem; margin-top: 0.5rem;'>Prediction: {display_result}</div>
                            </div>
                        """, unsafe_allow_html=True)

                except Exception as e:
                    st.warning(f"Could not plot EEG dynamically. Showing fallback image.\n\nDetails: {e}")
                    st.image("https://upload.wikimedia.org/wikipedia/commons/6/6c/EEG_Brainwaves.svg", caption="EEG Signal", use_column_width=True)
            else:
                st.image("https://upload.wikimedia.org/wikipedia/commons/6/6c/EEG_Brainwaves.svg", caption="EEG Signal", use_column_width=True)


 # spacing before buttons
    st.markdown("<br><br>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.link_button("Explanation", url="#")
    with col4:
        st.link_button("Download Report", url="#")


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
            st.write(f"Debug - Type: {type(uploaded_mri_file)}")
            st.write(f"Debug - Has name: {hasattr(uploaded_mri_file, 'name')}")
            if hasattr(uploaded_mri_file, 'name'):
                st.write(f"Debug - Name: {uploaded_mri_file.name}")
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
st.markdown("</div>", unsafe_allow_html=True)


st.markdown("""
<hr style="margin-top: 2rem; margin-bottom: 1rem;">
<div style="text-align: center; color: gray; font-size: 0.9rem;">
    © 2025 NeuroCheck • Developed by NeuroCheck
</div>
""", unsafe_allow_html=True)
