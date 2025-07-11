import streamlit as st
import pandas as pd
import numpy as np
import time
from model_utils import (
    load_model_by_name,
    predict_and_evaluate,
    plot_prediction,
    reshape_for_clstm,
    reshape_for_bpnn, 
    preprocess_raw_csv,
    summarize_model_scores,
    generate_comparison_conclusion
)

st.set_page_config(page_title="ANTMcast - Prediksi Saham ANTM", layout="centered")

# ----------------- HEADER ------------------
st.markdown("<h1 style='text-align:center;'>📈 ANTMcast</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Prediksi Harga Saham ANTM menggunakan <b>CLSTM</b> dan <b>BPNN</b></p>", unsafe_allow_html=True)
st.markdown("---")

# ----------------- RESET HANDLER ------------------
if "reset_ts" not in st.session_state:
    st.session_state.reset_ts = time.time()

if st.button("🔄 Reset Semua"):
    st.session_state.clear()
    st.session_state.reset_ts = time.time()
    st.rerun()

# ----------------- MODEL SELECTION ------------------
st.subheader("⚙️ Pilih Model Prediksi")
model_options = {
    "CLSTM - ANTM + Eksternal (IHSG, Emas, USD/IDR)": "clstm_antm_external",
    "BPNN - ANTM + Eksternal (IHSG, Emas, USD/IDR)": "bpnn_antm_external"
}
model_labels = {v: k for k, v in model_options.items()}
model_label = st.selectbox("Pilih model:", list(model_options.keys()))
model_name = model_options[model_label]

# ----------------- FILE UPLOAD ------------------
st.subheader("📄 Upload Data Mentah Saham ANTM")
raw_data = st.file_uploader(
    "Unggah file mentah dari Investing.com", 
    type="csv", 
    key=f"uploader_raw_{st.session_state.reset_ts}"
)

# ----------------- PREDIKSI TUNGGAL ------------------
if st.button("🔮 Jalankan Prediksi"):
    if raw_data is not None:
        try:
            df_raw = pd.read_csv(raw_data)
            use_external = "external" in model_name

            if use_external:
                df_ihsg = pd.read_csv("data/eksternal/ihsg_cleaned.csv", parse_dates=["Date"], index_col="Date")
                df_emas = pd.read_csv("data/eksternal/emas_cleaned.csv", parse_dates=["Date"], index_col="Date")
                df_usd = pd.read_csv("data/eksternal/usd_cleaned.csv", parse_dates=["Date"], index_col="Date")
                X_test, y_test = preprocess_raw_csv(df_raw, use_external=True,
                                                    df_ihsg=df_ihsg, df_emas=df_emas, df_usd=df_usd,
                                                    window_size=7)
            else:
                X_test, y_test = preprocess_raw_csv(df_raw, use_external=False, window_size=7)

            model = load_model_by_name(model_name)
            if model_name.startswith("clstm"):
                X_test = reshape_for_clstm(X_test)
            elif model_name.startswith("bpnn"):
                X_test = reshape_for_bpnn(X_test)

            result = predict_and_evaluate(model, X_test, y_test)

            st.success("✅ Prediksi berhasil!")
            st.markdown("### 📊 Hasil Evaluasi")
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", f"{result['mae']:.6f}")
            col2.metric("MSE", f"{result['mse']:.6f}")
            col3.metric("RMSE", f"{result['rmse']:.6f}")

            st.markdown("### 📉 Grafik Prediksi vs Aktual")
            fig = plot_prediction(y_test, result["y_pred"], title=f"Model: {model_labels[model_name].upper()}")
            st.pyplot(fig)

            st.session_state[f"X_test_{model_name}"] = X_test
            st.session_state[f"y_test_{model_name}"] = y_test
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan: {e}")
    else:
        st.warning("📎 Harap unggah file data saham terlebih dahulu.")

# ----------------- PERBANDINGAN SEMUA MODEL ------------------
if st.button("📋 Bandingkan Semua Model"):  
    score_dict = {}
    missing = []

    for name in model_options.values():
        X_key, y_key = f"X_test_{name}", f"y_test_{name}"
        if X_key in st.session_state and y_key in st.session_state:
            model = load_model_by_name(name)
            X = st.session_state[X_key]
            y = st.session_state[y_key]
            X_input = reshape_for_clstm(X) if name.startswith("clstm") else reshape_for_bpnn(X)
            try:
                result = predict_and_evaluate(model, X_input, y)
                score_dict[name] = result
            except:
                missing.append(name)
        else:
            missing.append(name)

    if score_dict:
        if missing:
            readable = [model_labels.get(name, name) for name in missing]
            st.warning("⚠️ Model berikut belum diprediksi dan dilewati: " + ", ".join(readable))

        df_summary = pd.DataFrame([{
            "Model": model_labels.get(name, name),
            "MAE": score["mae"],
            "MSE": score["mse"],
            "RMSE": score["rmse"]
        } for name, score in score_dict.items()])
        df_summary = df_summary.sort_values("MAE").reset_index(drop=True)

        st.markdown("### 📋 Tabel Perbandingan Model")
        st.dataframe(df_summary.style.format({"MAE": "{:.4f}", "MSE": "{:.4f}", "RMSE": "{:.4f}"}))

        csv = df_summary.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download CSV", data=csv, file_name="evaluasi_model_antm.csv", mime="text/csv")

        st.markdown("### 📉 Grafik Perbandingan Prediksi")
        for name in score_dict:
            fig = plot_prediction(
                st.session_state[f"y_test_{name}"],
                score_dict[name]["y_pred"],
                title=f"{model_labels.get(name, name).upper()} - Prediksi vs Aktual"
            )
            st.pyplot(fig)

        # 🔹 Tampilkan kesimpulan sederhana
        st.markdown("### 📝 Penjelasan & Kesimpulan Sederhana")
        conclusion = generate_comparison_conclusion(score_dict)
        st.info(conclusion)

    else:
        st.warning("📎 Belum ada model yang berhasil diprediksi untuk dibandingkan.")