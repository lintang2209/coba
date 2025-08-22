import streamlit as st
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from ultralytics import YOLO

# ====================
# CSS Styling dengan URL daun online
# ====================
st.markdown("""
    <style>
        .main {
            background-color: #f9fafb;
        }

        /* Daun kiri atas */
        body::before {
            content: "";
            position: absolute;
            top: -30px;
            left: -50px;
            width: 250px;
            height: 250px;
            background: url("https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.piqsels.com%2Fid%2Fpublic-domain-photo-siimx&psig=AOvVaw3RhuCGbjSlLn9fGVkNReX6&ust=1755967082523000&source=images&cd=vfe&opi=89978449&ved=0CBUQjRxqFwoTCMif0IDtno8DFQAAAAAdAAAAABAE") no-repeat;
            background-size: contain;
            transform: rotate(20deg);
            z-index: -1;
        }

        /* Daun kanan bawah */
        body::after {
            content: "";
            position: absolute;
            bottom: -30px;
            right: -50px;
            width: 250px;
            height: 250px;
            background: url("https://i.ibb.co/Z2ShYDC/leaf-bottom.png") no-repeat;
            background-size: contain;
            transform: rotate(-15deg);
            z-index: -1;
        }

        .center {
            text-align: center;
            padding-top: 120px;
        }

        .title {
            font-size: 36px;
            font-weight: 700;
            color: #4b8b64;
        }

        .subtitle {
            font-size: 16px;
            font-style: italic;
            color: #7d7d7d;
            margin-top: -10px;
        }

        .stButton>button {
            background-color: #f0f0f0;
            color: #4b8b64;
            border-radius: 20px;
            border: none;
            padding: 10px 25px;
            font-weight: 600;
            cursor: pointer;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #4b8b64;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# ====================
# Navigasi sederhana
# ====================
if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    # Landing page
    st.markdown("""
    <div class="center">
        <p class="title">ayo cek tanamanmu!</p>
        <p class="subtitle">kenali soybean rust sejak dini<br>untuk hasil panen yang lebih baik</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("cek disini"):
        st.session_state.page = "deteksi"
        st.experimental_rerun()

# ====================
# Halaman Deteksi (CNN & YOLO)
# ====================
elif st.session_state.page == "deteksi":

    st.title("Perbandingan Deteksi Penyakit Soybean Rust (CNN vs YOLO) 🌱")
    st.write("Unggah satu gambar daun kedelai untuk melihat hasil deteksi dari kedua model secara bersamaan.")

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

    # Muat model
    cnn_model = load_cnn_model()
    yolo_model = load_yolo_model()

    if cnn_model is None or yolo_model is None:
        st.stop()

    uploaded_file = st.file_uploader("Pilih gambar daun...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)
        st.write("---")

        col1, col2 = st.columns(2)

        # ==== CNN ====
        with col1:
            st.header("Hasil Analisis CNN")
            try:
                img_resized = image.resize((224, 224))
                img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

                prediction = cnn_model.predict(img_array)
                class_id = np.argmax(prediction)
                confidence = np.max(prediction)
                
                class_names = ["Daun Sehat", "Soybean Rust"]
                predicted_class_name = class_names[class_id]

                st.write(f"### Prediksi: **{predicted_class_name}**")
                st.write(f"Confidence: **{confidence:.2f}**")
            except Exception as e:
                st.error(f"Terjadi kesalahan pada model CNN: {e}")

        # ==== YOLOv8 ====
        with col2:
            st.header("Hasil Analisis YOLOv8")
            try:
                results = yolo_model(image)
                results_img = results[0].plot()
                st.image(results_img, caption="Hasil Deteksi YOLOv8", use_column_width=True)

                if len(results[0].boxes) > 0:
                    st.write("#### Detail Deteksi:")
                    for box in results[0].boxes:
                        conf = float(box.conf[0])
                        st.write(f"- Ditemukan **Penyakit Soybean Rust** dengan confidence **{conf:.2f}**")
                else:
                    st.write("Tidak ditemukan penyakit Soybean Rust.")
            except Exception as e:
                st.error(f"Terjadi kesalahan pada model YOLOv8: {e}")

    if st.button("⬅️ Kembali ke Beranda"):
        st.session_state.page = "home"
        st.experimental_rerun()
