import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# ============================================================
# ICONS (Lucide SVG Icons)
# ============================================================
ICONS = {
    "activity": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"/></svg>',
    "upload": '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/></svg>',
    "image": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/></svg>',
    "check_circle": '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>',
    "alert_circle": '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y1="16" y2="16"/></svg>',
    "alert_triangle": '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>',
    "info": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>',
    "shield": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/></svg>',
}

# Condition-specific styling
CONDITION_STYLES = {
    "Normal": {"color": "#059669", "bg": "#ecfdf5", "icon": "check_circle", "label": "Healthy"},
    "Cyst": {"color": "#2563eb", "bg": "#eff6ff", "icon": "info", "label": "Cyst Detected"},
    "Stone": {"color": "#d97706", "bg": "#fffbeb", "icon": "alert_circle", "label": "Stone Detected"},
    "Tumor": {"color": "#dc2626", "bg": "#fef2f2", "icon": "alert_triangle", "label": "Tumor Detected"},
}

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Kidney CT Analysis",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><text y='20' font-size='20'>⚕</text></svg>",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CSS STYLES
# ============================================================
st.markdown("""
<style>
    /* CSS Variables */
    :root {
        --primary: #1e40af;
        --primary-light: #3b82f6;
        --success: #059669;
        --warning: #d97706;
        --danger: #dc2626;
        --neutral-50: #fafafa;
        --neutral-100: #f5f5f5;
        --neutral-200: #e5e5e5;
        --neutral-300: #d4d4d4;
        --neutral-500: #737373;
        --neutral-600: #525252;
        --neutral-800: #262626;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 800px;
    }

    /* Header */
    .app-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
        border-bottom: 1px solid var(--neutral-200);
        margin-bottom: 2rem;
    }

    .app-header-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 48px;
        height: 48px;
        background: var(--primary);
        border-radius: 12px;
        margin-bottom: 0.75rem;
        color: white;
    }

    .app-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--neutral-800);
        margin: 0 0 0.25rem 0;
        letter-spacing: -0.025em;
    }

    .app-subtitle {
        font-size: 0.938rem;
        color: var(--neutral-500);
        margin: 0;
    }

    /* Upload area */
    .upload-area {
        border: 2px dashed var(--neutral-300);
        border-radius: 12px;
        padding: 2.5rem 1.5rem;
        text-align: center;
        background: var(--neutral-50);
        transition: border-color 0.2s, background-color 0.2s;
    }

    .upload-area:hover {
        border-color: var(--primary-light);
        background: white;
    }

    .upload-icon {
        color: var(--neutral-500);
        margin-bottom: 1rem;
    }

    .upload-text {
        font-size: 1rem;
        font-weight: 500;
        color: var(--neutral-800);
        margin: 0 0 0.25rem 0;
    }

    .upload-hint {
        font-size: 0.813rem;
        color: var(--neutral-500);
        margin: 0;
    }

    /* Section title */
    .section-title {
        font-size: 0.813rem;
        font-weight: 600;
        color: var(--neutral-500);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0 0 1rem 0;
    }

    /* Image card */
    .image-card {
        background: white;
        border: 1px solid var(--neutral-200);
        border-radius: 12px;
        overflow: hidden;
    }

    .image-card img {
        width: 100%;
        display: block;
    }

    .image-info {
        padding: 0.75rem 1rem;
        border-top: 1px solid var(--neutral-200);
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: var(--neutral-600);
        font-size: 0.813rem;
    }

    /* Diagnosis card */
    .diagnosis-card {
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }

    .diagnosis-icon {
        margin-bottom: 0.5rem;
    }

    .diagnosis-label {
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0 0 0.25rem 0;
        opacity: 0.8;
    }

    .diagnosis-result {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0 0 0.75rem 0;
    }

    .confidence-badge {
        display: inline-block;
        padding: 0.375rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
        background: rgba(255,255,255,0.5);
    }

    /* Probability section */
    .prob-section {
        background: white;
        border: 1px solid var(--neutral-200);
        border-radius: 12px;
        padding: 1.25rem;
        margin-top: 1.5rem;
    }

    .prob-title {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--neutral-800);
        margin: 0 0 1rem 0;
    }

    .prob-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.75rem;
    }

    .prob-item:last-child {
        margin-bottom: 0;
    }

    .prob-label {
        width: 60px;
        font-size: 0.813rem;
        font-weight: 500;
        color: var(--neutral-600);
    }

    .prob-bar-container {
        flex: 1;
        height: 8px;
        background: var(--neutral-100);
        border-radius: 4px;
        overflow: hidden;
    }

    .prob-bar {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }

    .prob-value {
        width: 45px;
        text-align: right;
        font-size: 0.813rem;
        font-weight: 600;
        color: var(--neutral-800);
    }

    /* Disclaimer */
    .disclaimer {
        margin-top: 2rem;
        padding: 1rem;
        background: var(--neutral-50);
        border-radius: 8px;
        border-left: 3px solid var(--warning);
    }

    .disclaimer-title {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.813rem;
        font-weight: 600;
        color: var(--neutral-800);
        margin: 0 0 0.5rem 0;
    }

    .disclaimer-text {
        font-size: 0.813rem;
        color: var(--neutral-600);
        margin: 0;
        line-height: 1.5;
    }

    /* Footer */
    .app-footer {
        margin-top: 2rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--neutral-200);
        text-align: center;
    }

    .footer-text {
        font-size: 0.75rem;
        color: var(--neutral-500);
        margin: 0;
    }

    /* Hide default file uploader label */
    .stFileUploader > label {
        display: none;
    }

    /* Style file uploader */
    .stFileUploader > div {
        padding: 0;
    }

    /* Analysis button */
    .stButton > button {
        width: 100%;
        background: var(--primary);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border-radius: 8px;
        transition: background 0.2s;
    }

    .stButton > button:hover {
        background: var(--primary-light);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown(f"""
<div class="app-header">
    <div class="app-header-icon">
        {ICONS['activity']}
    </div>
    <h1 class="app-title">Kidney CT Analysis</h1>
    <p class="app-subtitle">AI-Powered Diagnostic Screening Tool</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
MODEL_PATH = "model/kidney_model.h5"

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please ensure `kidney_model.h5` exists in the `model/` folder.")
    st.stop()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()
CLASS_NAMES = ["Cyst", "Normal", "Stone", "Tumor"]

# ============================================================
# PREPROCESS FUNCTION
# ============================================================
def preprocess_image(image):
    img = image.resize((96, 96))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# ============================================================
# MAIN UI
# ============================================================
uploaded_file = st.file_uploader(
    "Upload CT Scan",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

if not uploaded_file:
    # Show upload prompt
    st.markdown(f"""
    <div class="upload-area">
        <div class="upload-icon">
            {ICONS['upload']}
        </div>
        <p class="upload-text">Drop your CT scan image here</p>
        <p class="upload-hint">or click Browse files above</p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")

    # Create two columns for layout
    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.markdown('<p class="section-title">Uploaded Scan</p>', unsafe_allow_html=True)
        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown(f"""
            <div class="image-info">
                {ICONS['image']}
                <span>{uploaded_file.name}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="section-title">Analysis Result</p>', unsafe_allow_html=True)

        # Run prediction
        with st.spinner("Analyzing..."):
            img_array = preprocess_image(image)
            prediction = model.predict(img_array, verbose=0)[0]
            idx = np.argmax(prediction)
            result = CLASS_NAMES[idx]
            confidence = prediction[idx] * 100

        # Get styling for the result
        style = CONDITION_STYLES[result]

        # Display diagnosis card
        st.markdown(f"""
        <div class="diagnosis-card" style="background: {style['bg']}; color: {style['color']};">
            <div class="diagnosis-icon">
                {ICONS[style['icon']]}
            </div>
            <p class="diagnosis-label">{style['label']}</p>
            <h2 class="diagnosis-result">{result}</h2>
            <span class="confidence-badge">{confidence:.1f}% confidence</span>
        </div>
        """, unsafe_allow_html=True)

    # Probability breakdown
    st.markdown("""
    <div class="prob-section">
        <h3 class="prob-title">Probability Distribution</h3>
    """, unsafe_allow_html=True)

    # Create probability bars
    prob_colors = {
        "Cyst": "#2563eb",
        "Normal": "#059669",
        "Stone": "#d97706",
        "Tumor": "#dc2626"
    }

    prob_html = ""
    for i, class_name in enumerate(CLASS_NAMES):
        prob = prediction[i] * 100
        color = prob_colors[class_name]
        prob_html += f"""
        <div class="prob-item">
            <span class="prob-label">{class_name}</span>
            <div class="prob-bar-container">
                <div class="prob-bar" style="width: {prob}%; background: {color};"></div>
            </div>
            <span class="prob-value">{prob:.1f}%</span>
        </div>
        """

    st.markdown(prob_html + "</div>", unsafe_allow_html=True)

    # Disclaimer
    st.markdown(f"""
    <div class="disclaimer">
        <p class="disclaimer-title">
            {ICONS['shield']}
            Medical Disclaimer
        </p>
        <p class="disclaimer-text">
            This tool is intended for screening purposes only and should not be used as a definitive diagnosis.
            Always consult with a qualified healthcare professional for proper medical evaluation and treatment decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="app-footer">
    <p class="footer-text">
        Powered by MobileNetV2 · Built with TensorFlow & Streamlit
    </p>
</div>
""", unsafe_allow_html=True)
