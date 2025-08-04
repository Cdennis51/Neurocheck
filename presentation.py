import streamlit as st

st.set_page_config(page_title="Neurocheck App", layout="centered")

# Slide 1
st.title("What is Neurocheck?")
st.markdown("""
**Neurocheck** is a simple web app that helps:
- Detect **fatigue** from brain recordings (EEG)
- Detect **Alzheimer's** markers from brain scans (MRI)

Simply upload a file :file_folder: and get an instant prediction ⚡
""")

st.divider()

# Slide 2
st.header(":question: Why Does It Matter?")
st.markdown("""
Fatigue can negatively impact:

- Productivity at work 💼
- Focus and safety 🚗
- Mental and physical health ❤️

**Neurocheck** gives fast and accessible feedback to help you stay informed.
""")

st.divider()

# Slide 3
st.header("🖥️ How It Works (Simplified)")
st.markdown("""
1. You upload a brainwave file (`.csv` or `.edf`) 📁
2. The app connects to a backend (when online) 🌐
3. You receive a prediction like:
   - **Fatigued** or **Not Fatigued**
   - Confidence score (e.g. 87%)
""")

st.info("If the backend is offline, you'll see demo results.")

st.divider()

# Slide 4
st.header("🛠️ Features")
st.markdown("""
- Easy drag-and-drop interface
- Accepts `.CSV` and `.EDF` files (up to 200MB)
- Shows demo results if backend is unavailable
- Built using **Streamlit** — no installation needed
""")

st.divider()

# Slide 5
st.header("🔄 What’s Coming Next")
st.markdown("""
- Real-time predictions from a live backend
- More file types and advanced feedback
- Personalized fatigue history and reports
""")

st.divider()

# Slide 6
st.header("👥 Who Is It For?")
st.markdown("""
- Students and researchers 🧪
- Health and wellness enthusiasts 🧘
- Anyone curious about their mental state 🧍
- No technical knowledge required!
""")

st.divider()

# Slide 7
st.header("✅ Summary")
st.markdown("""
- **Neurocheck** makes EEG fatigue detection accessible
- Easy to use and fully online
- Still improving — feedback is welcome! 🙌
""")

st.success("Thank you!")
