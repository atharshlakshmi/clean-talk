import requests
import streamlit as st

prompt = st.text_input("Enter your prompt:")
if st.button("Classify"):
    response = requests.post("http://localhost:8000/classify", json={"prompt": prompt})
    result = response.json()
    st.write(f"Classification: {result['classification']}")
    st.write(f"Confidence: {result['confidence']}")