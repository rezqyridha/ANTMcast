# preprocess_eksternal.py

import pandas as pd
import numpy as np
import os

RAW_PATH = "data/raw/"
SAVE_PATH = "data/eksternal/"
FILES = {
    "ihsg": "ihsg_10y.csv",
    "emas": "emas_10y.csv",
    "usd": "usd_idr_10y.csv"
}

def clean_external_file(filepath, name=""):
    print(f"\n🔄 Mulai proses: {name.upper()} - {filepath}")

    df = pd.read_csv(filepath)

    # 🔁 Cek apakah hanya ada 1 kolom (format quote semua → butuh split manual)
    if df.shape[1] == 1:
        print("📎 Format hanya 1 kolom — parsing ulang...")
        try:
            df = pd.read_csv(filepath, sep=",", quotechar='"')
            if df.shape[1] == 1:
                df = df[df.columns[0]].str.split(",", expand=True)
                df.columns = df.iloc[0]
                df = df[1:]
        except Exception as e:
            raise ValueError(f"❌ Gagal parsing ulang CSV: {e}")

    # 🧾 Log header awal
    print("🧾 Kolom awal:", df.columns.tolist())

    # 🧹 Bersihkan header
    df.columns = df.columns.str.replace('"', '').str.strip()
    df.rename(columns=lambda x: x.replace('"', '').strip(), inplace=True)

    # 🔁 Rename 'Price' → 'Close'
    if 'Price' in df.columns:
        df.rename(columns={'Price': 'Close'}, inplace=True)

    # 📅 Bersihkan kolom Date secara eksplisit
    if 'Date' not in df.columns:
        raise ValueError("❌ Kolom 'Date' tidak ditemukan dalam file.")
    df['Date'] = df['Date'].astype(str).str.replace('"', '').str.strip()

    # 🔢 Bersihkan angka: koma → float
    for col in ['Close', 'Open', 'High', 'Low']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", "").str.replace('"', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 🔄 Volume
    if 'Vol.' in df.columns:
        df['Vol.'] = df['Vol.'].astype(str).str.replace('"', '')
        df['Vol.'] = df['Vol.'].replace({'K': '*1e3', 'M': '*1e6', 'B': '*1e9'}, regex=True)
        df['Vol.'] = df['Vol.'].apply(lambda x: pd.eval(x) if isinstance(x, str) and x.strip() not in ['nan', ''] else np.nan)
        df.rename(columns={'Vol.': 'Volume'}, inplace=True)

    # 🔄 Change %
    if 'Change %' in df.columns:
        df['Change %'] = df['Change %'].astype(str).str.replace('%', '').str.replace('"', '')
        df['Change %'] = pd.to_numeric(df['Change %'], errors='coerce')
        df.rename(columns={'Change %': '%Change'}, inplace=True)

    # 🔄 Konversi tanggal dan sort
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values("Date")

    # 🔢 Simpan hanya kolom numerik
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_cleaned = df[['Date'] + numeric_cols].copy()
    df_cleaned = df_cleaned.set_index("Date")

    print("✅ Selesai — contoh:\n", df_cleaned.head(3))
    return df_cleaned

# Proses dan simpan
for key, filename in FILES.items():
    try:
        df_clean = clean_external_file(os.path.join(RAW_PATH, filename), name=key)
        save_path = os.path.join(SAVE_PATH, f"{key}_cleaned.csv")
        df_clean.to_csv(save_path)
        print(f"✅ Disimpan ke: {save_path}")
    except Exception as e:
        print(f"❌ Gagal proses {key.upper()}: {e}")
