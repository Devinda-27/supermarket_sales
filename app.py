import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Konfigurasi Judul Halaman
st.set_page_config(page_title="Supermarket Sales Clustering")
st.title("📊 Supermarket Sales Clustering App")
st.write("Aplikasi ini mengelompokkan data penjualan berdasarkan Gross Income dan Customer Rating.")

# 1. Load Data
@st.cache_data # Menggunakan cache agar data tidak di-load ulang setiap kali slider digeser
def load_data():
    url = "https://raw.githubusercontent.com/Devinda-27/supermarket_sales/6d2f7be684e43ec8cad547e4ccf94130dc67cbc8/supermarket_Sales.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# Tampilkan data sekilas
if st.checkbox("Tampilkan Raw Data"):
    st.write(df.head())

# 2. Preprocessing
X = df[['Gross income', 'Customer stratification rating']]

# 3. Sidebar untuk Input Parameter
st.sidebar.header("Konfigurasi K-Means")
k_value = st.sidebar.slider("Pilih jumlah Cluster (K):", min_value=2, max_value=10, value=5)

# 4. Model K-Means
kmeans = KMeans(n_clusters=k_value, random_state=0)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# 5. Visualisasi
st.subheader(f"Hasil Clustering dengan K={k_value}")
fig, ax = plt.subplots(figsize=(10, 6))

# Plot data points
scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis', alpha=0.6)
# Plot centroids
ax.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Centroids')

ax.set_xlabel('Gross Income')
ax.set_ylabel('Customer Stratification Rating')
ax.set_title('K-Means Clustering Visualisation')
ax.legend()

# Tampilkan plot di Streamlit
st.pyplot(fig)

# 6. Analisis Tambahan (Inertia)
st.divider()
st.write("Nilai Inertia saat ini:", round(kmeans.inertia_, 2))
