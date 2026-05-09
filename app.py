import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Title
st.title("Clustering Pelanggan")

st.write("Analisis laba kotor pelanggan dan rating pelanggan")

# Upload dataset
uploaded_file = st.file_uploader("Upload Dataset CSV", type=["csv"])

if uploaded_file is not None:
    
    df = pd.read_csv(uploaded_file)

    st.write("Dataset")
    st.dataframe(df)

    # Ambil fitur
    X = df[['gross income', 'Rating']]

    # Prediksi cluster
    cluster = model.predict(X)

    df['Cluster'] = cluster

    st.write("Hasil Clustering")
    st.dataframe(df)

    # Visualisasi
    fig, ax = plt.subplots()

    scatter = ax.scatter(
        df['gross income'],
        df['Rating'],
        c=df['Cluster'],
        cmap='viridis'
    )

    ax.set_xlabel("Gross Income")
    ax.set_ylabel("Rating")

    st.pyplot(fig)
