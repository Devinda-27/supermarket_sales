import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans

# =====================================
# LOAD DATASET
# =====================================
df = pd.read_csv("Mall_Customers.csv")

# Ambil fitur
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# =====================================
# TRAINING MODEL
# =====================================
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# =====================================
# SIMPAN MODEL
# =====================================
with open("model.pkl", "wb") as file:
    pickle.dump(kmeans, file)

# =====================================
# LOAD MODEL
# =====================================
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# =====================================
# STREAMLIT APP
# =====================================
st.set_page_config(page_title="Clustering Customer")

st.title("Aplikasi Clustering Customer")
st.write("Masukkan data customer untuk menentukan cluster.")

# Input user
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

# Tombol prediksi
if st.button("Prediksi Cluster"):

    # Data input
    data = np.array([[income, score]])

    # Prediksi
    hasil = model.predict(data)

    # Tampilkan hasil
    st.success(f"Customer masuk ke Cluster {hasil[0]}")
