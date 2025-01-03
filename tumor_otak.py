import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
import os
import pandas as pd  # Untuk visualisasi dan download data

# ==========================================
# Load Model (tanpa konfigurasi GPU)
# ==========================================
@st.cache_resource  # Agar model di-cache dan tidak dimuat berkali-kali
def load_cnn_model():
    model_path = 'tumor_otak.h5'  # Sesuaikan dengan path model Anda
    if not os.path.exists(model_path):
        st.error(f"Gagal memuat model: File tidak ditemukan di path '{model_path}'")
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
        # Resize gambar ke (224, 224)
        img = img.resize((224, 224))
        
        # Pastikan gambar diubah ke RGB untuk memastikan 3 channel
        img = img.convert("RGB")
        
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
        
        return label, prob_percent, preds
    
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
        return None, None, None

# ==========================================
# Streamlit App
# ==========================================

# ==========================================
# 1. Tambahkan Navigasi ke Website Utama
# ==========================================
st.sidebar.title("Navigasi")
main_website_url = "https://k11-cnn-detection.vercel.app/"  # Ganti dengan URL website utama Anda
st.sidebar.markdown(f"[🔙 Kembali ke Website Utama]({main_website_url})")

# ==========================================
# 2. Judul dan Deskripsi Aplikasi
# ==========================================
st.title("Brain Tumor Detection App")
st.write("""
Aplikasi ini menggunakan model CNN untuk mendeteksi apakah gambar MRI 
termasuk dalam kategori **Glioma**, **Meningioma**, **No Tumor**, atau **Pituitary**.
Silakan upload gambar MRI di bawah ini untuk deteksi.
""")

# ==========================================
# 3. Widget File Uploader
# ==========================================
uploaded_file = st.file_uploader("Upload Gambar MRI", type=["png", "jpg", "jpeg"])

# ==========================================
# 4. Tampilkan Gambar, Prediksi, dan Fitur Tambahan
# ==========================================
if uploaded_file is not None:
    try:
        # Baca file sebagai PIL Image
        image_pil = Image.open(uploaded_file)
        
        # Tampilkan gambar yang di-upload
        st.image(image_pil, caption="Gambar yang di-upload", use_column_width=True)
        
        # Tombol prediksi
        if st.button("Prediksi"):
            with st.spinner("Melakukan prediksi..."):
                label, prob, preds = predict_brain_tumor(image_pil)
            
            if label is not None:
                # Menampilkan hasil prediksi
                st.success(f"Hasil Prediksi: **{label}**")
                st.info(f"Probabilitas: **{prob:.2f}%**")
                
                # -------------------------------------------
                # 4.1 Menambahkan Visualisasi Probabilitas
                # -------------------------------------------
                st.write("Probabilitas untuk setiap kelas:")
                prob_dict = CLASS_LABELS
                prob_values = preds[0] * 100
                prob_labels = [CLASS_LABELS[k] for k in prob_dict.keys()]
                
                # Membuat DataFrame untuk visualisasi
                df_probs = pd.DataFrame({
                    'Kelas': prob_labels,
                    'Probabilitas (%)': prob_values
                }).set_index('Kelas')
                
                # Menampilkan grafik batang
                st.bar_chart(df_probs)
                
                # -------------------------------------------
                # 4.2 Menambahkan Opsi Download Hasil Prediksi
                # -------------------------------------------
                st.write("Anda dapat mendownload hasil prediksi dalam bentuk file CSV.")
                
                # Membuat DataFrame untuk hasil prediksi
                df_result = pd.DataFrame({
                    'Label': [label],
                    'Probabilitas (%)': [prob]
                })
                
                # Tombol download
                st.download_button(
                    label="Download Hasil Prediksi",
                    data=df_result.to_csv(index=False),
                    file_name='hasil_prediksi.csv',
                    mime='text/csv',
                )
    except Exception as e:
        st.error(f"Gagal memproses gambar: {e}")
