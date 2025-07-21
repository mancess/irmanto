import streamlit as st
import pickle

# Load model dan vectorizer
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.title("Analisis Sentimen Ulasan Produk E-Commerce")

ulasan = st.text_area("Masukkan ulasan produk:")

if st.button("Prediksi Sentimen"):
    if ulasan.strip():
        vectorized = vectorizer.transform([ulasan])
        prediksi = model.predict(vectorized)[0]
        st.success(f"Sentimen: {prediksi.capitalize()}")
    else:
        st.warning("Masukkan ulasan terlebih dahulu.")