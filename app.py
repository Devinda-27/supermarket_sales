import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("Mall_Customers.csv")

# Ambil fitur
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# =========================
# TRAINING MODEL
# =========================
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)

# =========================
# SIMPAN MODEL
# =========================
with open('model.pkl', 'wb') as file:
    pickle.dump(kmeans, file)

# =========================
# LOAD MODEL
# =========================
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# =========================
# STREAMLIT APP
# =========================
st.title("Aplikasi Clustering Customer")

st.write("Masukkan data customer")

# Input user
income = st.number_input(
    "Annual Income (k$)",
    min_value=0
)

score = st.number_input(
    "Spending Score (1-100)",
    min_value=0,
    max_value=100
)

# Tombol prediksi
if st.button("Prediksi Cluster"):

    # Ubah ke array
    data = np.array([[income, score]])

    # Prediksi cluster
    cluster = model.predict(data)

    st.success(f"Customer masuk ke Cluster {cluster[0]}")
