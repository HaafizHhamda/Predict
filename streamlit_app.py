import os, json
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Predict review")

st.title("Tulis review")
st.write("Review anda sebuah movitasi bagi kami")

# ====== KONFIG (ubah sesuai nama file kamu) ======
MODEL_PATH = "model_lstm_1.keras"   # letakkan di folder yang sama dengan app.py
CLASS_JSON = "class_names.json"     # opsional; kalau ada, dipakai

@st.cache_resource(show_spinner=False)
def load_infer_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")
    return load_model(MODEL_PATH, compile=False)

@st.cache_resource(show_spinner=False)
def load_labels(n_fallback: int):
    if os.path.exists(CLASS_JSON):
        try:
            with open(CLASS_JSON, "r", encoding="utf-8") as f:
                m = json.load(f)  # {"0":"1 star", ...}
            # ubah ke list berurut 0..K-1
            keys = sorted(m.keys(), key=lambda x: int(x))
            return [m[k] for k in keys]
        except:
            pass
    # fallback sederhana 1..n
    return [f"{i+1} star" if i == 0 else f"{i+1} stars" for i in range(n_fallback)]

def ensure_2d_string_tensor(texts):
    arr = np.array(texts, dtype=object).reshape(-1, 1)
    return tf.constant(arr, dtype=tf.string)

def to_probs(raw):
    raw = np.array(raw)
    if raw.ndim == 1: raw = raw[:, None]
    if (raw < 0).any() or not np.allclose(raw.sum(axis=1), 1.0, atol=1e-3):
        x = raw - np.max(raw, axis=-1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=-1, keepdims=True)
    return raw

# ====== UI paling simpel ======
txt = st.text_area("", height=140, placeholder="Mcdonald is the best!")
predict = st.button("Predict", type="primary")

# ====== Aksi ======
if predict:
    try:
        model = load_infer_model()
        X = ensure_2d_string_tensor([txt.strip()])
        raw = model.predict(X, verbose=0)
        probs = to_probs(raw)

        labels = load_labels(probs.shape[1])
        pred_idx = int(np.argmax(probs, axis=1)[0])
        st.success(f"Predicted: {labels[pred_idx]}")
    except Exception as e:
        st.error(f"Gagal prediksi: {e}")
