import streamlit as st
from model import predict_genre

st.set_page_config(page_title="Movie Genre Classifier")

st.title("🎬 Movie Genre Classification")
st.write("Predict movie genre using Machine Learning")

plot = st.text_area("Enter movie description")

if st.button("Predict Genre"):
    if plot.strip():
        genre = predict_genre(plot)
        st.success(f"Predicted Genre: {genre}")
    else:
        st.warning("Please enter a description")
