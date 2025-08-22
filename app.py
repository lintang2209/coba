import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import torch
import os

# --- TITLE & DESCRIPTION ---
st.title("Aplikasi Deteksi Penyakit Soybean Rust")
st.write("Aplikasi ini menggunakan dua model berbeda (CNN dan YOLOv8) untuk mendeteksi penyakit Soybean Rust pada daun kedelai. Silakan unggah gambar daun kedelai untuk mendapatkan hasil deteksi.")

# --- LOAD MODELS ---
# Pastikan file model Anda ada di direktori yang sama
@st.cache_resource
def load_cnn_model():
    # Ganti 'cnn_model.h5' dengan nama file model CNN Anda
    model = tf.keras.models.load_model('models/cnn_soybean_rust.keras')
    return model

@st.cache_resource
def load_yolo_model():
    # Ganti 'yolov8_model.pt' dengan nama file model YOLOv8 Anda
    model = YOLO('models/best.pt')
    return model

# --- LOAD CLASS NAMES ---
# Ganti dengan nama kelas yang sesuai dengan model Anda
CLASS_NAMES_CNN = ['Healthy', 'Soybean Rust']
CLASS_NAMES_YOLO = ['Soybean Rust']

# --- SIDEBAR & MODEL SELECTION ---
with st.sidebar:
    st.header("Pengaturan Model")
    selected_model = st.radio("Pilih Model untuk Deteksi", ["CNN", "YOLOv8"])
    
    if selected_model == "CNN":
        model = load_cnn_model()
        class_names = CLASS_NAMES_CNN
        st.write("Model yang dipilih: **CNN**")
    else:
        model = load_yolo_model()
        class_names = CLASS_NAMES_YOLO
        st.write("Model yang dipilih: **YOLOv8**")

# --- IMAGE UPLOADER ---
uploaded_file = st.file_uploader("Unggah gambar daun kedelai...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Buka gambar dan tampilkan
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Gambar yang Diunggah.', use_column_width=True)
    st.write("")
    st.write("Menganalisis gambar...")

    # --- PREDICTION LOGIC ---
    if selected_model == "CNN":
        try:
            # Resize gambar untuk model CNN (sesuaikan dengan ukuran input model Anda)
            image_resized = image.resize((224, 224))
            img_array = np.array(image_resized) / 255.0  # Normalisasi
            img_array = np.expand_dims(img_array, axis=0)
            
            # Prediksi
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions)
            predicted_class_name = class_names[predicted_class_index]
            confidence = np.max(predictions) * 100
            
            st.success(f"Hasil Deteksi: **{predicted_class_name}**")
            st.write(f"Tingkat Keyakinan: **{confidence:.2f}%**")
            
        except Exception as e:
            st.error(f"Terjadi kesalahan saat menjalankan model CNN: {e}")

    elif selected_model == "YOLOv8":
        try:
            # Prediksi dengan YOLOv8
            results = model.predict(source=image, conf=0.25, verbose=False)
            
            # Tampilkan hasil deteksi
            for r in results:
                # Dapatkan gambar dengan kotak deteksi
                im_bgr = r.plot()
                im_rgb = Image.fromarray(im_bgr[..., ::-1])
                
                # Tampilkan gambar
                st.image(im_rgb, caption='Hasil Deteksi YOLOv8', use_column_width=True)

                # Tampilkan jumlah deteksi
                if len(r.boxes) > 0:
                    st.success(f"Ditemukan **{len(r.boxes)}** objek penyakit Soybean Rust.")
                else:
                    st.success("Tidak ditemukan penyakit Soybean Rust.")
                    
        except Exception as e:
            st.error(f"Terjadi kesalahan saat menjalankan model YOLOv8: {e}")
            st.write("Pastikan model YOLOv8 Anda kompatibel dan tidak ada masalah saat prediksi.")
