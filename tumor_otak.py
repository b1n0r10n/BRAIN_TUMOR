import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
import os

# ==========================================
# Load Model (tanpa konfigurasi GPU)
# ==========================================
@st.cache_resource  # Agar model di-cache dan tidak dimuat berkali-kali
def load_cnn_model():
    model_path = 'tumor_otak.h5'  # Sesuaikan dengan path model Anda
    if not os.path.exists(model_path):
        st.error(f"Gagal memuat model: File tidak ditemukan di path {model_path}")
        return None
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

model = load_cnn_model()

# Pastikan model berhasil dimuat sebelum melanjutkan
if model is None:
    st.stop()

# ==========================================
# Mapping Label Kelas
# ==========================================
CLASS_LABELS = {
    0: 'Glioma',
    1: 'Meningioma',
    2: 'No Tumor',
    3: 'Pituitary'
}

# ==========================================
# Fungsi Prediksi
# ==========================================
def predict_brain_tumor(img: Image.Image):
    """
    Menerima input berupa objek PIL Image, 
    kemudian melakukan prediksi kelas tumor otak.
    """
    try:
        # Pastikan gambar diubah ke RGB untuk memastikan 3 channel
        img = img.convert("RGB")

        # Resize gambar ke (224, 224)
        img = img.resize((224, 224))
        
        # Convert gambar menjadi array numpy
        img_array = image.img_to_array(img)
        
        # Expand dim agar shape menjadi (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocessing sesuai dengan VGG16 (atau model lain yang serupa)
        img_array = preprocess_input(img_array)
        
        # Verifikasi bentuk array
        if img_array.shape != (1, 224, 224, 3):
            raise ValueError(f"Ukuran input tidak sesuai: {img_array.shape}. Harus (1, 224, 224, 3)")
        
        # Prediksi
        preds = model.predict(img_array)
        
        # Ambil index prediksi tertinggi
        predicted_class = np.argmax(preds, axis=1)[0]
        
        # Label & probabilitas
        label = CLASS_LABELS.get(predicted_class, "Unknown")
        prob_percent = preds[0][predicted_class] * 100
        
        return label, prob_percent
    
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
        return None, None

# ==========================================
# Streamlit App
# ==========================================
st.title("Brain Tumor Detection App")
st.write("""
Aplikasi ini menggunakan model CNN untuk deteksi apakah gambar MRI 
termasuk kategori **Glioma**, **Meningioma**, **No Tumor**, atau **Pituitary**.
Silakan upload gambar MRI di bawah ini untuk deteksi.
""")

# Widget file_uploader untuk mengunggah gambar
uploaded_file = st.file_uploader("Upload Gambar MRI", type=["png", "jpg", "jpeg"])

# Tombol untuk melakukan prediksi
if uploaded_file is not None:
    try:
        # Baca file sebagai PIL Image
        image_pil = Image.open(uploaded_file)

        # Tampilkan gambar yang diupload
        st.image(image_pil, caption="Gambar yang di-upload", use_column_width=True)

        # Tombol prediksi
        if st.button("Prediksi"):
            with st.spinner("Melakukan prediksi..."):
                label, prob = predict_brain_tumor(image_pil)
            
            if label is not None:
                st.success(f"Hasil Prediksi: **{label}**")
                st.info(f"Probabilitas: **{prob:.2f}%**")

        # Tambahkan tombol untuk kembali ke website utama
        st.markdown("""
            <a href="https://www.website-utama-anda.com" target="_self">
                <button style="
                    background-color:#4CAF50; 
                    color:white; 
                    padding:10px 20px; 
                    border:none; 
                    border-radius:5px; 
                    cursor:pointer;
                    font-size:16px;
                    margin-top:20px;
                ">
                    Kembali ke Website Utama
                </button>
            </a>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Gagal memproses gambar: {e}")
