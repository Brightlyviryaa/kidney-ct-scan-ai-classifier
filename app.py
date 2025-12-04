import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="AI Kidney CT Classifier",
    page_icon="🧬",
    layout="centered",
)

# ======================================================
# CUSTOM CSS FOR PREMIUM UI
# ======================================================
st.markdown("""
<style>
    body {
        background: linear-gradient(135deg, #d7e1ec, #fefefe);
    }

    .main-title {
        font-size: 40px;
        font-weight: 900;
        text-align: center;
        background: -webkit-linear-gradient(#2b5876, #4e4376);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: -10px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #5a5a5a;
        margin-bottom: 30px;
    }

    .upload-box {
        border: 2.5px dashed #4e4376;
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        color: #4e4376;
        font-weight: 600;
        background: rgba(255, 255, 255, 0.45);
        backdrop-filter: blur(8px);
        transition: 0.3s;
    }
    .upload-box:hover {
        border-color: #2b5876;
        transform: scale(1.02);
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.65);
        padding: 25px;
        border-radius: 25px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        animation: fadeIn 1s ease-in-out;
    }

    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(15px);}
        to {opacity: 1; transform: translateY(0);}
    }

    .result-title {
        font-size: 28px;
        font-weight: 800;
        text-align: center;
        color: #2b5876;
    }

    .confidence-text {
        text-align: center;
        font-size: 20px;
        margin-top: -10px;
        color: #4e4376;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD MODEL
# ======================================================
MODEL_PATH = "model/kidney_model.h5"

st.markdown("<h1 class='main-title'>🧬 Kidney CT Scan AI Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Early Detection • Medical Imaging • AI-Powered Diagnosis</p>", unsafe_allow_html=True)

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model tidak ditemukan! Pastikan file `kidney_model.h5` ada di folder `model/`.")
    st.stop()

model = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = ["Cyst", "Normal", "Stone", "Tumor"]

# ======================================================
# PREPROCESS
# ======================================================
def preprocess_image(image):
    img = image.resize((96, 96))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# ======================================================
# FILE UPLOAD
# ======================================================
uploaded_file = st.file_uploader("📤 Upload CT Scan Image", type=["jpg", "jpeg", "png"])

if not uploaded_file:
    st.markdown("<div class='upload-box'>✨ Seret atau pilih file CT Scan di sini ✨</div>", unsafe_allow_html=True)

else:
    image = Image.open(uploaded_file).convert("RGB")

    # Show image
    st.image(image, width=350, caption="📸 CT Scan Uploaded", output_format="JPEG")

    # Predict
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)[0]
    idx = np.argmax(prediction)
    result = CLASS_NAMES[idx]
    confidence = prediction[idx] * 100

    # ======================================================
    # RESULT CARD (Glassmorphism)
    # ======================================================
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    st.markdown(f"<div class='result-title'>🧪 Diagnosis: <b>{result}</b></div>", unsafe_allow_html=True)
    st.markdown(f"<p class='confidence-text'>Confidence Level: <b>{confidence:.2f}%</b></p>", unsafe_allow_html=True)

    st.progress(float(confidence / 100))

    st.markdown("</div>", unsafe_allow_html=True)

    # ======================================================
    # PROBABILITY (Horizontal bar chart)
    # ======================================================
    st.write("### 📊 Probability Breakdown")

    prob_dict = {
        "Cyst": float(prediction[0]),
        "Normal": float(prediction[1]),
        "Stone": float(prediction[2]),
        "Tumor": float(prediction[3]),
    }

    st.bar_chart(prob_dict)

