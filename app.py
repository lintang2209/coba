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
    MODEL_PATH = "cnn_soybean_rust.keras"
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

@st.cache_resource
def load_yolo_model():
    MODEL_PATH = "yolov8_soybean_rust.pt"
    model = YOLO(MODEL_PATH)
    return model

# ==== PILIH MODE ====
USE_MODEL = "YOLO"  # "CNN" atau "YOLO"

if USE_MODEL == "CNN":
    model = load_cnn_model()
    class_names = ["Daun Sehat", "Soybean Rust"]
elif USE_MODEL == "YOLO":
    model = load_yolo_model()

uploaded_file = st.file_uploader("Pilih gambar daun...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_container_width=True)

    if USE_MODEL == "CNN":
        img_resized = image.resize((224, 224))
        img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

        prediction = model.predict(img_array)
        class_id = np.argmax(prediction)
        confidence = np.max(prediction)

        st.write(f"### Prediksi: {class_names[class_id]}")
        st.write(f"Confidence: {confidence:.2f}")

    elif USE_MODEL == "YOLO":
        results = model(image)
        results_img = results[0].plot()
        st.image(results_img, caption="Hasil Deteksi YOLOv8", use_container_width=True)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                st.write(f"Deteksi: {r.names[cls_id]} (confidence {conf:.2f})")
