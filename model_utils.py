# model_utils.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# 🔹 1. Load model .keras berdasarkan nama file
def load_model_by_name(name: str):
    path = f"models/{name}.keras"
    return load_model(path)

# 🔹 2. Evaluasi dan Prediksi
def predict_and_evaluate(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        return {
            "y_pred": y_pred.flatten(),
            "mae": mae,
            "mse": mse,
            "rmse": rmse
        }
    except Exception as e:
        raise ValueError(f"❌ Terjadi kesalahan saat prediksi: {e}")

# 🔹 3. Visualisasi Hasil Prediksi
def plot_prediction(y_test, y_pred, title="Prediksi vs Aktual"):
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, label='Aktual')
    plt.plot(y_pred, label='Prediksi', linestyle='--')
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Harga Saham (Normalized)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt

# 🔹 4. Ringkasan Performa Semua Model (dipertahankan versi ini saja)
def summarize_model_scores(model_score_dict: dict, round_digits=6, sort_by="MAE"):
    """
    Membuat dataframe ringkasan skor MAE, MSE, RMSE dari banyak model.
    """
    df = pd.DataFrame({
        name: {
            "MAE": score["mae"],
            "MSE": score["mse"],
            "RMSE": score["rmse"]
        }
        for name, score in model_score_dict.items()
    }).T
    if sort_by in df.columns:
        df = df.sort_values(sort_by)
    return df.round(round_digits)


# 🔹 5. Reshape untuk CLSTM (from 2D to 3D)
def reshape_for_clstm(X: np.ndarray):
    if len(X.shape) == 2:
        return np.expand_dims(X, axis=-1)
    return X

# 🔹 6. Reshape untuk BPNN (from 3D to 2D)
def reshape_for_bpnn(X: np.ndarray):
    if len(X.shape) == 3:
        return X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
    return X


# 🔹 7. Safe Prediction
def safe_predict(model, X):
    try:
        return model.predict(X)
    except Exception as e:
        raise ValueError(f"❌ Input tidak cocok dengan model: {e}")

# 🔹 8. Preprocessing file mentah dari Investing.com
def preprocess_raw_csv(df, use_external=False, df_ihsg=None, df_emas=None, df_usd=None, window_size=10):

    # 🟡 Parsing ulang jika hanya 1 kolom
    if df.shape[1] == 1:
        try:
            df = pd.read_csv(df.columns[0], sep=",", quotechar='"')
        except Exception:
            raise ValueError("❌ Gagal parsing ulang file CSV mentah. Pastikan format sesuai dari Investing.com")

    # 🧹 Bersihkan kolom
    df.columns = df.columns.str.replace('"', '').str.strip()
    df = df.rename(columns=lambda x: x.replace('"', '').strip())

    # 🔁 Price → Close
    if 'Price' in df.columns:
        df.rename(columns={'Price': 'Close'}, inplace=True)

    # 🧼 Bersihkan Volume
    if 'Vol.' in df.columns:
        df['Vol.'] = df['Vol.'].astype(str).str.replace('"', '')
        df['Vol.'] = df['Vol.'].replace({'K': '*1e3', 'M': '*1e6', 'B': '*1e9'}, regex=True)
        df['Vol.'] = df['Vol.'].apply(lambda x: pd.eval(x) if isinstance(x, str) and x.strip() not in ['nan', ''] else np.nan)
        df.rename(columns={'Vol.': 'Volume'}, inplace=True)

    # 🧼 Bersihkan %Change
    if 'Change %' in df.columns:
        df['Change %'] = df['Change %'].astype(str).str.replace('%', '', regex=False)
        df['Change %'] = pd.to_numeric(df['Change %'], errors='coerce')
        df.rename(columns={'Change %': '%Change'}, inplace=True)

    # 🗓️ Konversi dan sort tanggal
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values("Date")

    # ❗ Validasi kolom target
    if 'Close' not in df.columns:
        raise ValueError("❌ Kolom 'Close' tidak ditemukan meskipun sudah mencoba mengganti dari 'Price'")

    # 🧼 Bersihkan koma di angka
    for col in ['Close', 'Open', 'High', 'Low']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", "").astype(float)

    # Gunakan index tanggal
    df.set_index("Date", inplace=True)

    # 🔍 Tentukan kolom yang akan digunakan untuk pelatihan
    if use_external:
        # Hanya Close ANTM + eksternal nanti
        df_main = df[['Close']].copy()
    else:
        # Gunakan 6 fitur ANTM saja
        fitur_antm = ['Open', 'High', 'Low', 'Close', 'Volume', '%Change']
        missing = [col for col in fitur_antm if col not in df.columns]
        if missing:
            raise ValueError(f"❌ Kolom berikut tidak ditemukan dalam data ANTM: {missing}")
        df_main = df[fitur_antm].copy()

    # 🔗 Gabungkan eksternal jika diminta
    if use_external and all([df_ihsg is not None, df_emas is not None, df_usd is not None]):
        eksternal_data = {'ihsg': df_ihsg, 'emas': df_emas, 'usd': df_usd}
        for name, ext_df in eksternal_data.items():
            print(f"\n📦 Debug {name.upper()} — Kolom awal: {ext_df.columns.tolist()}")

            if ext_df.index.name != 'Date':
                if 'Date' not in ext_df.columns:
                    raise ValueError(f"❌ Kolom 'Date' tidak ditemukan di eksternal: {name}")
                ext_df['Date'] = pd.to_datetime(ext_df['Date'], errors='coerce')
                ext_df = ext_df.dropna(subset=['Date']).sort_values("Date")
                ext_df.set_index("Date", inplace=True)

            if 'Close' not in ext_df.columns:
                raise ValueError(f"❌ Kolom 'Close' tidak tersedia di data eksternal: {name}")

            # Gabung hanya kolom Close
            df_main = df_main.join(ext_df[['Close']], how='inner', rsuffix=f"_{name}")
            print(f"📊 Gabung {name.upper()} selesai. Kolom sekarang: {df_main.columns.tolist()}")

    # ✅ Log kolom akhir
    print("✅ Kolom akhir untuk training:", df_main.columns.tolist())

    # 🔄 Normalisasi
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df_main)

    # ❗ Validasi jumlah data cukup
    if len(df_scaled) <= window_size:
        raise ValueError(f"❌ Data terlalu sedikit untuk window size = {window_size}. Minimal {window_size+1} baris, tersedia: {len(df_scaled)}")

    if window_size <= 0:
        raise ValueError("❌ Ukuran window harus lebih besar dari 0")

    # 🪜 Sliding window
    X, y = [], []
    close_idx = df_main.columns.get_loc("Close")
    for i in range(window_size, len(df_scaled)):
        X.append(df_scaled[i - window_size:i])
        y.append(df_scaled[i][close_idx])

    return np.array(X), np.array(y)


# 🔹 9. Ekspor skor evaluasi ke CSV
def export_scores_to_csv(df, filename="evaluasi_model_antm.csv"):
    return df.to_csv(index=True).encode('utf-8'), filename

# 🔹 10. Buat ringkasan narasi sederhana
def generate_comparison_conclusion(score_dict):
    """
    Membuat narasi kesimpulan perbandingan CLSTM + eksternal dan BPNN + eksternal
    dengan format rapi satu baris per model, mudah dipahami orang awam.
    """
    lines = []

    # Ringkasan per model (nama disederhanakan)
    for model_name, result in score_dict.items():
        if "CLSTM" in model_name.upper():
            readable_name = "CLSTM + eksternal"
        elif "BPNN" in model_name.upper():
            readable_name = "BPNN + eksternal"
        else:
            readable_name = model_name.upper()

        mae_percent = result['mae'] * 100  # ubah ke persen agar familiar
        lines.append(
            f"📌 Model **{readable_name}** memiliki MAE sekitar {mae_percent:.2f}%, "
            f"MSE {result['mse']:.4f}, dan RMSE {result['rmse']:.4f}.\n"
        )

    # Penjelasan interpretasi metrik
    lines.append(
        "✅ **Interpretasi:**\n"
        "- Nilai MAE menunjukkan rata-rata seberapa besar prediksi meleset dari harga saham asli.\n"
        "- Semakin kecil MAE dan RMSE, semakin akurat model memprediksi harga saham.\n"
    )

    # Penjelasan grafik
    lines.append(
        "📈 **Penjelasan Grafik:**\n"
        "- Garis biru menampilkan harga saham asli (aktual).\n"
        "- Garis oranye putus-putus menampilkan prediksi model.\n"
        "- Pola kedua garis yang rapat dan searah menunjukkan prediksi mengikuti tren data asli.\n"
        "- Walaupun data pengguna bisa berbeda-beda rentang waktunya, pola ini tetap valid sebagai pembanding.\n"
    )

    # Kesimpulan awam
    lines.append(
        "📝 **Kesimpulan:**\n"
        "Model dengan nilai MAE dan RMSE lebih kecil, serta grafik prediksi yang pola naik-turunnya mirip dengan data asli, "
        "lebih baik dipilih untuk memprediksi harga saham ANTM."
    )

    return "\n".join(lines)
