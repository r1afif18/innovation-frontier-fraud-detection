# ==============================================================================
# APLIKASI STREAMLIT UNTUK DETEKSI PENIPUAN LINGUISTIK
# ==============================================================================

import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
import os

# --- KONFIGURASI HALAMAN ---
# Mengatur konfigurasi dasar halaman web, seperti judul di tab browser dan ikon.
st.set_page_config(
    page_title="Detektor Penipuan Linguistik",
    page_icon="🔎",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- FUNGSI & PEMUATAN MODEL ---

# Menggunakan cache Streamlit agar model tidak perlu dimuat ulang setiap kali ada interaksi.
# Ini membuat aplikasi jauh lebih cepat.
@st.cache_resource
def load_assets():
    """Memuat model dan vectorizer yang telah dilatih dari folder saved_models."""
    try:
        model_path = os.path.join('saved_models', 'model.pkl')
        vectorizer_path = os.path.join('saved_models', 'tfidf_vectorizer.pkl')
        
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        # Memastikan stopwords NLTK tersedia di lingkungan deployment.
        try:
            stopwords.words('indonesian')
        except LookupError:
            nltk.download('stopwords')

        return model, vectorizer
    except FileNotFoundError:
        # Menampilkan pesan error yang jelas jika file model tidak ditemukan.
        st.error("Error: File model atau vectorizer tidak ditemukan. Pastikan file 'model.pkl' dan 'tfidf_vectorizer.pkl' ada di dalam folder 'saved_models/'.")
        return None, None

# Fungsi pra-pemrosesan teks.
# PENTING: Fungsi ini harus 100% identik dengan yang digunakan saat melatih model.
def preprocess_text(text, stopwords_list):
    """Membersihkan dan memproses teks input dari pengguna."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join(word for word in text.split() if word not in stopwords_list and len(word) > 2)
    return text.strip()

# Memuat aset (model & vectorizer) saat aplikasi pertama kali dijalankan.
model, tfidf_vectorizer = load_assets()
if model and tfidf_vectorizer:
    # Mempersiapkan daftar stopwords untuk digunakan oleh fungsi preprocess_text.
    stopwords_indonesia = stopwords.words('indonesian')
    stopwords_indonesia.extend(['yg', 'dg', 'dgn', 'dr', 'kpd', 'utk', 'kak', 'gan', 'sis'])

# --- ANTARMUKA PENGGUNA (UI) STREAMLIT ---

# Judul dan Deskripsi
st.title("🔎 Detektor Sidik Jari Linguistik Penipuan")
st.markdown("""
Aplikasi ini menggunakan model *machine learning* untuk menganalisis gaya bahasa sebuah pesan dan memprediksi apakah pesan tersebut berpotensi sebagai penipuan. Cukup masukkan teks pesan di bawah dan klik tombol "Analisis Pesan".
""")

st.warning("**Disclaimer:** Model ini adalah alat bantu prediktif berdasarkan pola teks dan **bukan bukti hukum**. Selalu verifikasi informasi secara mandiri sebelum melakukan tindakan apa pun.")

# Area Input Teks dari Pengguna
user_input = st.text_area("Masukkan teks pesan yang ingin dianalisis:", height=150, placeholder="Contoh: Selamat! Nomor Anda memenangkan hadiah 100jt, silakan hubungi admin...")

# Tombol untuk Memicu Analisis
if st.button("Analisis Pesan", type="primary"):
    if user_input and model and tfidf_vectorizer:
        # Proses hanya jika ada input dan model berhasil dimuat
        
        # 1. Terapkan pra-pemrosesan pada input pengguna
        clean_input = preprocess_text(user_input, stopwords_indonesia)
        
        # 2. Ubah teks bersih menjadi vektor numerik menggunakan TF-IDF
        input_vector = tfidf_vectorizer.transform([clean_input])
        
        # 3. Lakukan prediksi dan dapatkan probabilitasnya
        prediction = model.predict(input_vector)[0]
        prediction_proba = model.predict_proba(input_vector)
        
        # Menampilkan hasil prediksi utama
        st.subheader("Hasil Analisis:")
        
        col1, col2 = st.columns([0.6, 0.4]) # Mengatur lebar kolom
        
        with col1:
            if prediction == 1:
                st.error("⚠️ **TERDETEKSI PENIPUAN**")
            else:
                st.success("✅ **TERDETEKSI BUKAN PENIPUAN**")
        
        with col2:
            # Menampilkan skor risiko sebagai metrik yang jelas
            risk_score = prediction_proba[0][1] * 100
            st.metric(label="Skor Risiko Penipuan", value=f"{risk_score:.2f}%")

        # Memberikan visualisasi bar untuk skor risiko
        st.progress(int(risk_score / 100))
        
        # Memberikan penjelasan kontekstual berdasarkan skor risiko
        if risk_score > 75:
            st.write("Interpretasi: Model sangat yakin pesan ini adalah penipuan. Harap berhati-hati dan jangan memberikan informasi pribadi atau melakukan transfer.")
        elif risk_score > 40:
            st.write("Interpretasi: Model mendeteksi beberapa ciri yang mirip dengan pola penipuan. Sebaiknya waspada dan jangan mengklik tautan sembarangan.")
        else:
            st.write("Interpretasi: Model tidak menemukan pola linguistik kuat yang mengindikasikan penipuan dalam pesan ini.")

    elif not user_input:
        st.warning("Silakan masukkan teks untuk dianalisis.")
    # Jika model gagal dimuat, pesan error akan ditampilkan oleh fungsi load_assets

# Menambahkan informasi tambahan di bagian bawah menggunakan tab
st.write("---")
st.subheader("Informasi Tambahan")

tabs = st.tabs(["Bagaimana Cara Kerjanya?", "Performa Model", "Tentang Proyek"])

with tabs[0]:
    st.markdown("""
    Aplikasi ini bekerja dalam beberapa langkah:
    1.  **Pra-pemrosesan Teks:** Teks yang Anda masukkan akan dibersihkan dari tanda baca, angka, dan kata-kata umum (stopwords).
    2.  **Ekstraksi Fitur (TF-IDF):** Teks yang sudah bersih diubah menjadi representasi numerik yang dapat dipahami oleh model. Model akan mengukur pentingnya setiap kata dalam teks Anda dibandingkan dengan ribuan contoh lain yang telah dipelajari.
    3.  **Klasifikasi (Logistic Regression):** Vektor numerik tersebut kemudian dimasukkan ke dalam model klasifikasi yang telah dilatih pada lebih dari seribu contoh pesan penipuan dan bukan penipuan.
    4.  **Skor Risiko:** Model menghasilkan probabilitas yang kami ubah menjadi skor 0-100 untuk menunjukkan tingkat kepercayaan model terhadap prediksinya.
    """)

with tabs[1]:
    st.markdown("""
    Model ini dievaluasi menggunakan metrik yang relevan untuk kasus deteksi penipuan. Berikut adalah contoh performa model pada data uji:
    - **F1-Score (Kelas Penipuan):** 0.92 (Sangat baik dalam menyeimbangkan antara menemukan kasus penipuan dan tidak salah menuduh).
    - **Recall (Kelas Penipuan):** 0.95 (Artinya, model berhasil mengidentifikasi 95% dari seluruh kasus penipuan yang ada di data uji).
    - **Precision (Kelas Penipuan):** 0.89 (Artinya, dari semua yang diprediksi sebagai penipuan, 89% di antaranya memang benar-benar penipuan).
    
   
    """)
   

with tabs[2]:
    st.markdown("""
    Proyek ini adalah bagian dari studi kasus **"Innovation Frontiers"** yang bertujuan untuk menerapkan ilmu data pada masalah-masalah sosial yang kompleks dan "anti-mainstream" di Indonesia.
    
    - **Dibuat oleh:** Rafif Sudanta
    - **Lihat Kode:** [GitHub Repository](https://github.com/r1afif18/innovation-frontier-fraud-detection/)
    """)
