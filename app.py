import streamlit as st
import numpy as np
from PIL import Image
import keras
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Fruit Ripeness Detector", page_icon="", layout="centered")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    model_path = os.path.join("models", "fruit_ripeness_model.h5")
    if not os.path.exists(model_path):
        st.error(f"Model tidak ditemukan di path: {model_path}")
        return None
    try:
        return keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        return None

model = load_model()
class_names = ['ripe', 'rotten', 'unripe']

# --- FUNGSI PREDIKSI ---
def predict_image(model, image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    return predicted_class, confidence

# --- UI UTAMA ---
st.markdown("<h1 style='text-align: center;'>Fruit Scanner</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>AI Ripeness Detector</p>", unsafe_allow_html=True)

# Input Method
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
with col2:
    camera_input = st.camera_input("Use Camera")

# Pilih sumber gambar
image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
elif camera_input is not None:
    image = Image.open(camera_input)

# Tampilkan Hasil Jika Ada Gambar
if image is not None and model is not None:
    st.image(image, width=700)
    
    pred_class, confidence = predict_image(model, image)
    
    # Custom Circular Progress Bar dengan SVG
    radius = 50
    circumference = 2 * np.pi * radius
    stroke_dashoffset = circumference - (confidence / 100) * circumference
    
    svg_circle = f"""
    <div style="display: flex; justify-content: center; align-items: center; margin: 20px 0;">
        <svg width="120" height="120" viewBox="0 0 120 120">
            <circle cx="60" cy="60" r="{radius}" fill="none" stroke="#e0e0e0" stroke-width="10"/>
            <circle cx="60" cy="60" r="{radius}" fill="none" stroke="#2ecc71" stroke-width="10" 
                    stroke-dasharray="{circumference}" stroke-dashoffset="{stroke_dashoffset}" 
                    transform="rotate(-90 60 60)" stroke-linecap="round"/>
            <text x="60" y="65" font-size="24" font-weight="bold" text-anchor="middle" fill="#333">{int(confidence)}%</text>
        </svg>
    </div>
    """
    st.markdown(svg_circle, unsafe_allow_html=True)
    
    # Status & Confidence Text
    status_color = {"ripe": "#2ecc71", "rotten": "#e74c3c", "unripe": "#f39c12"}
    color = status_color.get(pred_class, "#333")
    
    st.markdown(f"<h3 style='text-align: center; color: {color};'>Status: {pred_class.upper()}</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center;'>Confidence: <b>{confidence:.2f}%</b></p>", unsafe_allow_html=True)
    
    # Smart Tags & Rekomendasi
    tags = []
    advice = ""
    if pred_class == 'ripe':
        tags = ["Sweet", "Soft Texture", "Ready to Eat"]
        advice = "Buah siap dikonsumsi! Nikmati sekarang."
    elif pred_class == 'unripe':
        tags = ["Hard Texture", "Sour Taste", "Wait 2-3 Days"]
        advice = "Simpan di suhu ruang selama 2-3 hari lagi."
    else:  # rotten
        tags = ["Overripe", "Bad Smell", "Do Not Eat"]
        advice = "Jangan dikonsumsi. Segera buang atau komposkan."
    
    # ✅ DIPERBAIKI: Keluar dari blok if-else, agar tags muncul untuk SEMUA kelas
    cols = st.columns(3)
    for i, tag in enumerate(tags):
        cols[i].markdown(
            f"""
            <div style='
                background: rgba(255, 255, 255, 0.1); 
                border: 1px solid rgba(255, 255, 255, 0.2); 
                padding: 10px; 
                border-radius: 20px; 
                text-align: center; 
                font-size: 14px; 
                color: white; 
                font-weight: 500;'>
                {tag}
            </div>
            """, 
            unsafe_allow_html=True
        )
    st.info(advice)

elif model is None:
    st.warning("⚠️ Model belum dimuat. Pastikan file 'fruit_ripeness_model.h5' ada di folder 'models/'.")