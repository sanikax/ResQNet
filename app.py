"""
ResQNet - Ambulance Detection System
app.py: Streamlit web application for live image prediction

Run with: streamlit run app.py

Author: [Your Name]
College: Savitribai Phule Pune University (SPPU)
"""

import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image
from model import IMG_HEIGHT, IMG_WIDTH

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ResQNet – Ambulance Detector",
    page_icon="🚑",
    layout="centered"
)

# ─────────────────────────────────────────────
# LOAD MODEL (cached for performance)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "models/resqnet_best.h5"
    if not os.path.exists(model_path):
        st.error("Model file not found! Train the model first using `python train.py`.")
        st.stop()
    return tf.keras.models.load_model(model_path)

model = load_model()
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# Class mapping — must match train_generator.class_indices from training
CLASS_NAMES = {0: "🚑 Ambulance", 1: "🚗 Non-Ambulance"}

# ─────────────────────────────────────────────
# UI — HEADER
# ─────────────────────────────────────────────
st.title("🚑 ResQNet — Ambulance Detection System")
st.markdown(
    "**A CNN-based emergency vehicle classifier**  \n"
    "Upload an image to detect whether it's an *Ambulance* or a *Regular Vehicle*."
)
st.divider()

# ─────────────────────────────────────────────
# UI — SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About ResQNet")
    st.write(
        "ResQNet uses a Convolutional Neural Network (CNN) trained on labeled "
        "vehicle images to classify emergency and non-emergency vehicles."
    )
    st.subheader("Model Architecture")
    st.code(
        "Input: 128×128×3\n"
        "Conv Block 1: 32 filters\n"
        "Conv Block 2: 64 filters\n"
        "Conv Block 3: 128 filters\n"
        "Dense: 256 → 128 → 1\n"
        "Output: Sigmoid (Binary)",
        language="text"
    )
    st.subheader("Threshold")
    threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05)

# ─────────────────────────────────────────────
# UI — IMAGE UPLOAD
# ─────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📂 Upload Vehicle Image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, PNG, WEBP"
)

if uploaded_file:
    col1, col2 = st.columns([1, 1])

    # Show uploaded image
    with col1:
        pil_img = Image.open(uploaded_file).convert("RGB")
        st.image(pil_img, caption="Uploaded Image", use_column_width=True)

    # Preprocess & predict
    img_resized  = pil_img.resize(IMG_SIZE)
    img_array    = keras_image.img_to_array(img_resized) / 255.0
    img_batch    = np.expand_dims(img_array, axis=0)

    with st.spinner("🔍 Analyzing image..."):
        raw_score = float(model.predict(img_batch, verbose=0)[0][0])

    # Determine class (verify class index mapping from training!)
    is_ambulance = raw_score < threshold
    confidence   = (1 - raw_score) if is_ambulance else raw_score
    label        = CLASS_NAMES[0] if is_ambulance else CLASS_NAMES[1]
    color        = "green" if is_ambulance else "red"

    # Show results
    with col2:
        st.subheader("Prediction Result")
        st.markdown(f"### :{color}[{label}]")
        st.metric("Confidence", f"{confidence*100:.1f}%")
        st.metric("Raw Score (sigmoid)", f"{raw_score:.4f}")

        st.progress(int(confidence * 100))

        if is_ambulance:
            st.success("✅ Emergency vehicle detected! Priority clearance recommended.")
        else:
            st.info("ℹ️ Regular vehicle — no emergency action needed.")

    st.divider()
    st.caption(
        f"Model: ResQNet CNN  |  Input size: {IMG_HEIGHT}×{IMG_WIDTH}  "
        f"|  Threshold: {threshold}"
    )

else:
    st.info("👆 Upload a vehicle image to get started.")
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/"
        "Ambulance_Mercedes_Benz_Sprinter.jpg/320px-Ambulance_Mercedes_Benz_Sprinter.jpg",
        caption="Example: Ambulance",
        width=300
    )
