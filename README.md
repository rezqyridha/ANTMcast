# 📈 ANTMcast - Prediksi Harga Saham ANTM Menggunakan CLSTM dan BPNN

**ANTMcast** adalah aplikasi berbasis _Streamlit_ untuk melakukan prediksi harga saham PT Aneka Tambang Tbk. (ANTM) dengan pendekatan _machine learning_ dan _deep learning_. Aplikasi ini membandingkan performa dua algoritma:

-   🧠 **CLSTM** (Convolutional Long Short-Term Memory)
-   🔁 **BPNN** (Backpropagation Neural Network)

---

## 🚀 Fitur Utama

-   📤 Upload file data saham ANTM (.csv) dari Investing.com
-   🔍 Jalankan prediksi harga saham menggunakan model CLSTM atau BPNN
-   📊 Evaluasi akurasi model menggunakan MAE, MSE, dan RMSE
-   📉 Visualisasi grafik prediksi vs harga aktual
-   📋 Bandingkan performa semua model dalam bentuk tabel dan grafik
-   ⬇️ Download hasil evaluasi model ke dalam format CSV

---

## 🧠 Model yang Digunakan

| Model                    | Input Fitur Utama                       |
| ------------------------ | --------------------------------------- |
| CLSTM - ANTM Saja        | Open, High, Low, Close, Volume, %Change |
| CLSTM - ANTM + Eksternal | Close ANTM + IHSG, Harga Emas, USD/IDR  |
| BPNN - ANTM Saja         | Sama seperti CLSTM ANTM                 |
| BPNN - ANTM + Eksternal  | Sama seperti CLSTM Eksternal            |

---

## 🗂️ Struktur Proyek

```
📦 ANTMcast/
├── app.py               # Aplikasi utama Streamlit
├── model_utils.py       # Fungsi bantu (preprocessing, evaluasi, dll)
├── models/              # Model .keras hasil pelatihan
├── data/                # Dataset eksternal: IHSG, Emas, USD/IDR
├── notebooks/           # Notebook pelatihan CLSTM & BPNN
├── requirements.txt     # Daftar library untuk deployment
└── README.md            # Dokumentasi proyek
```

---

## 💻 Jalankan Secara Lokal

1. Clone repository:

```bash
git clone https://github.com/username/ANTMcast.git
cd ANTMcast
```

2. Install dependensi:

```bash
pip install -r requirements.txt
```

3. Jalankan aplikasi:

```bash
streamlit run app.py
```

---
