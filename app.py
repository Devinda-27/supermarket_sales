import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ── Konfigurasi Halaman ──────────────────────────────────────────────────────
st.set_page_config(page_title="Supermarket Sales Clustering", page_icon="📊")
st.title("📊 Supermarket Sales Clustering App")
st.write("Aplikasi ini mengelompokkan data penjualan berdasarkan **Gross Income** dan **Customer Rating**, lalu memprediksi cluster untuk pelanggan baru.")

# ── 1. Load Data ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    url = (
        "https://raw.githubusercontent.com/Devinda-27/supermarket_sales/"
        "6d2f7be684e43ec8cad547e4ccf94130dc67cbc8/supermarket_Sales.csv"
    )
    df = pd.read_csv(url)
    return df

df = load_data()

if st.checkbox("Tampilkan Raw Data"):
    st.dataframe(df.head())

# ── 2. Preprocessing ─────────────────────────────────────────────────────────
gross_col  = "Gross income"
rating_col = "Customer stratification rating"
X = df[[gross_col, rating_col]].dropna()

# ── 3. Sidebar ───────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Konfigurasi K-Means")
k_value = st.sidebar.slider("Jumlah Cluster (K):", min_value=2, max_value=10, value=3)

st.sidebar.markdown("---")
st.sidebar.header("🔍 Prediksi Pelanggan Baru")

gross_input  = st.sidebar.number_input(
    "Gross Income ($):",
    min_value=0.0,
    max_value=float(X[gross_col].max() * 2),
    value=float(X[gross_col].mean()),
    step=0.01,
    format="%.2f",
)
rating_input = st.sidebar.number_input(
    "Customer Rating (1–10):",
    min_value=1.0,
    max_value=10.0,
    value=float(X[rating_col].mean()),
    step=0.1,
    format="%.1f",
)

predict_btn = st.sidebar.button("🚀 Prediksi Cluster", use_container_width=True)

# ── 4. Train K-Means ─────────────────────────────────────────────────────────
kmeans    = KMeans(n_clusters=k_value, random_state=42, n_init=10)
labels    = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# ── 5. Visualisasi Clustering ─────────────────────────────────────────────────
st.subheader(f"Hasil Clustering dengan K = {k_value}")

colors   = plt.cm.tab10.colors
fig, ax  = plt.subplots(figsize=(10, 6))

for c in range(k_value):
    mask = labels == c
    ax.scatter(
        X[gross_col][mask],
        X[rating_col][mask],
        color=colors[c % len(colors)],
        alpha=0.55,
        s=40,
        label=f"Cluster {c + 1}",
    )

ax.scatter(
    centroids[:, 0], centroids[:, 1],
    s=220, c="red", marker="X", zorder=5, label="Centroid",
)

# Tandai posisi input pelanggan jika tombol ditekan
if predict_btn:
    ax.scatter(
        gross_input, rating_input,
        s=300, c="gold", marker="*", edgecolors="black",
        zorder=6, linewidths=0.8, label="Pelanggan Anda",
    )

ax.set_xlabel("Gross Income ($)")
ax.set_ylabel("Customer Rating")
ax.set_title("K-Means Clustering — Supermarket Sales")
ax.legend(loc="upper right", fontsize=8)
st.pyplot(fig)

st.caption(f"Nilai Inertia: **{kmeans.inertia_:.2f}**")

# ── 6. Prediksi Cluster ───────────────────────────────────────────────────────
if predict_btn:
    st.markdown("---")
    st.subheader("🎯 Hasil Prediksi Pelanggan Baru")

    user_point    = np.array([[gross_input, rating_input]])
    user_cluster  = int(kmeans.predict(user_point)[0])
    centroid      = centroids[user_cluster]

    # Hitung jumlah anggota cluster
    cluster_size  = int((labels == user_cluster).sum())

    # Tentukan karakteristik cluster
    avg_income    = float(X[gross_col][labels == user_cluster].mean())
    avg_rating    = float(X[rating_col][labels == user_cluster].mean())
    income_level  = "rendah" if avg_income < 15 else ("sedang" if avg_income < 30 else "tinggi")
    rating_level  = "rendah" if avg_rating < 4 else ("cukup" if avg_rating < 7 else "tinggi")

    # Tampilkan hasil
    col1, col2, col3 = st.columns(3)
    col1.metric("🏷️ Cluster", f"Cluster {user_cluster + 1}")
    col2.metric("👥 Jumlah Anggota", cluster_size)
    col3.metric("📍 Jarak ke Centroid",
                f"{np.linalg.norm(user_point - centroid):.2f}")

    st.info(
        f"**Pelanggan Anda masuk ke Cluster {user_cluster + 1}.**\n\n"
        f"Kelompok ini memiliki rata-rata gross income **{income_level}** "
        f"(rata-rata **${avg_income:.2f}**) dan tingkat kepuasan pelanggan "
        f"**{rating_level}** (rata-rata **{avg_rating:.2f}**). "
        f"Terdapat **{cluster_size}** pelanggan dalam kelompok ini."
    )

    with st.expander("📋 Detail Centroid Cluster"):
        centroid_df = pd.DataFrame(
            centroids,
            columns=[gross_col, rating_col],
            index=[f"Cluster {i + 1}" for i in range(k_value)],
        )
        centroid_df.index.name = "Cluster"
        st.dataframe(centroid_df.style.highlight_min(axis=0, color="#ffd6d6")
                                      .highlight_max(axis=0, color="#d6ffd6")
                                      .format("{:.2f}"))
