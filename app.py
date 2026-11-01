"""
Deep Learning & AI Internship — Week 4 Capstone
Streamlit Web Dashboard for Medical Chest X-Ray Analysis
"""

import os
import streamlit as st
from PIL import Image
import requests

st.set_page_config(page_title="Medical Image Analysis Capstone", page_icon="🏥", layout="wide")

st.title("🏥 Medical Image Analysis: Pneumonia Detection Pipeline")
st.write("Deep Learning Diagnostic Support Tool using ResNet50 & Convolutional Neural Networks")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Patient Chest X-Ray")
    uploaded_file = st.file_uploader("Choose a Chest X-Ray Image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-Ray Scan", use_column_width=True)

        if st.button("🚀 Analyze X-Ray Scan"):
            with st.spinner("Processing image tensor through CNN..."):
                try:
                    bytes_data = uploaded_file.getvalue()
                    response = requests.post(
                        "http://127.0.0.1:8000/predict",
                        files={"file": (uploaded_file.name, bytes_data, uploaded_file.type)}
                    )
                    if response.status_code == 200:
                        res = response.json()
                        st.success(f"**Diagnosis Result:** {res['diagnosis']}")
                        st.info(f"**Model Confidence:** {res['confidence']}")
                    else:
                        st.error("API error during inference.")
                except Exception as e:
                    st.warning("FastAPI backend offline. Direct Python inference required.")

with col2:
    st.subheader("📊 Model Performance & Evaluation Metrics")
    tab1, tab2 = st.tabs(["Confusion Matrix", "ROC Curve"])

    with tab1:
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", caption="Confusion Matrix Evaluation")
        else:
            st.info("Run `python train_eval.py` to generate the Confusion Matrix.")

    with tab2:
        if os.path.exists("roc_curve.png"):
            st.image("roc_curve.png", caption="ROC Curve & AUC Metric")
        else:
            st.info("Run `python train_eval.py` to generate the ROC Curve.")