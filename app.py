import streamlit as st
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from ultralytics import YOLO

# ==== UI ====
st.title("Deteksi Penyakit Daun Kedelai 🌱")
st.write("Upload gambar daun kedelai untuk deteksi penyakit Soybean Rust.")

# ==== LOAD MODEL DENGAN CACHING ====
@st.cache_resource
def load_cnn_model():
    # Pastikan jalur file model sudah benar
    MODEL_PATH = "models/cnn_soybean_rust.keras"
    if not os.path.exists(MODEL_PATH):
        st.error(f"File model tidak ditemukan: {MODEL_PATH}")
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model CNN: {e}")
        return None

@st.cache_resource
def load_yolo_model():
    # Pastikan jalur file model sudah benar
    MODEL_PATH = "models/best.pt"
    if not os.path.exists(MODEL_PATH):
        st.error(f"File model tidak ditemukan: {MODEL_PATH}")
        return None
    try:
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model YOLO: {e}")
        return None

# ==== PILIH MODE ====
USE_MODEL = "YOLO" # "CNN" atau "YOLO"

if USE_MODEL == "CNN":
    model = load_cnn_model()
    class_names = ["Daun Sehat", "Soybean Rust"]
elif USE_MODEL == "YOLO":
    model = load_yolo_model()

# Handle jika model gagal dimuat
if model is None:
    st.stop()

uploaded_file = st.file_uploader("Pilih gambar daun...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    # Mengatasi TypeError: use_container_width
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    if USE_MODEL == "CNN":
        # Preprocessing sesuai arsitektur CNN
        img_resized = image.resize((224, 224))
        img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

        # Prediksi
        prediction = model.predict(img_array)
        class_id = np.argmax(prediction)
        confidence = np.max(prediction)

        st.write(f"### Prediksi: {class_names[class_id]}")
        st.write(f"Confidence: {confidence:.2f}")

    elif USE_MODEL == "YOLO":
        try:
            # Jalankan deteksi
            results = model(image)
            
            # Tampilkan hasil deteksi dengan bounding box
            results_img = results[0].plot()
            st.image(results_img, caption="Hasil Deteksi YOLOv8", use_column_width=True)

            # Ambil info kelas + confidence
            if len(results[0].boxes) > 0:
                st.write("### Hasil Deteksi:")
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    st.write(f"- Ditemukan **Penyakit Soybean Rust** dengan confidence: {conf:.2f}")
            else:
                st.write("### Hasil Deteksi:")
                st.write("Tidak ditemukan penyakit Soybean Rust.")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat menjalankan model YOLOv8: {e}")
