import streamlit as st
import pickle
import numpy as np

# ==========================
# LOAD MODEL
# ==========================
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# ==========================
# TAMPILAN STREAMLIT
# ==========================
st.set_page_config(page_title="Clustering Customer")

st.title("Aplikasi Clustering Customer")
st.write("Input data customer untuk menentukan cluster.")

# ==========================
# INPUT USER
# ==========================
income = st.number_input(
    "Annual Income (k$)",
    min_value=0,
    step=1
)

score = st.number_input(
    "Spending Score (1-100)",
    min_value=0,
    max_value=100,
    step=1
)

# ==========================
# PREDIKSI CLUSTER
# ==========================
if st.button("Prediksi Cluster"):

    # Ubah menjadi array
    data = np.array([[income, score]])

    # Prediksi cluster
    hasil = model.predict(data)

    # Tampilkan hasil
    st.success(f"Customer masuk ke Cluster {hasil[0]}")
