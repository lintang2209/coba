import streamlit as st
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from ultralytics import YOLO

# ==== UI ====
st.title("Perbandingan Deteksi Penyakit Soybean Rust (CNN vs YOLO) 🌱")
st.write("Unggah satu gambar daun kedelai untuk melihat hasil deteksi dari kedua model secara bersamaan.")

# ==== LOAD MODEL DENGAN CACHING ====
@st.cache_resource
def load_cnn_model():
    MODEL_PATH = "models/cnn_soybean_rust_new.h5"
    if not os.path.exists(MODEL_PATH):
        st.error(f"File model CNN tidak ditemukan: {MODEL_PATH}")
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model CNN: {e}")
        return None

@st.cache_resource
def load_yolo_model():
    MODEL_PATH = "models/best.pt"
    if not os.path.exists(MODEL_PATH):
        st.error(f"File model YOLOv8 tidak ditemukan: {MODEL_PATH}")
        return None
    try:
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model YOLOv8: {e}")
        return None

# Muat kedua model di awal aplikasi
cnn_model = load_cnn_model()
yolo_model = load_yolo_model()

# Jika salah satu model gagal dimuat, hentikan aplikasi
if cnn_model is None or yolo_model is None:
    st.stop()

uploaded_file = st.file_uploader("Pilih gambar daun...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)
    st.write("---")

    # Kolom untuk perbandingan hasil
    col1, col2 = st.columns(2)

    # ==== HASIL DETEKSI CNN ====
    with col1:
        st.header("Hasil Analisis CNN")
        try:
            # Preprocessing sesuai arsitektur CNN
            img_resized = image.resize((224, 224))
            img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

            # Prediksi
            prediction = cnn_model.predict(img_array)
            class_id = np.argmax(prediction)
            confidence = np.max(prediction)
            
            class_names = ["Daun Sehat", "Soybean Rust"]
            predicted_class_name = class_names[class_id]

            st.write(f"### Prediksi: **{predicted_class_name}**")
            st.write(f"Confidence: **{confidence:.2f}**")
        except Exception as e:
            st.error(f"Terjadi kesalahan pada model CNN: {e}")

    # ==== HASIL DETEKSI YOLOv8 ====
    with col2:
        st.header("Hasil Analisis YOLOv8")
        try:
            # Jalankan deteksi
            results = yolo_model(image)
            results_img = results[0].plot()
            st.image(results_img, caption="Hasil Deteksi YOLOv8", use_column_width=True)

            # Tampilkan info deteksi
            if len(results[0].boxes) > 0:
                st.write("#### Detail Deteksi:")
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    st.write(f"- Ditemukan **Penyakit Soybean Rust** dengan confidence **{conf:.2f}**")
            else:
                st.write("Tidak ditemukan penyakit Soybean Rust.")
        except Exception as e:
            st.error(f"Terjadi kesalahan pada model YOLOv8: {e}")
