
# Medical Image Analysis Capstone (Deep Learning & AI — Week 4 Major Project)

An end-to-end Deep Learning Capstone pipeline for automated Pneumonia classification from Chest X-Ray scans. Features dataset augmentation, ResNet50 CNN fine-tuning, performance evaluation (Confusion Matrix & ROC-AUC curves), a FastAPI backend, a Streamlit dashboard, and Docker deployment specs[cite: 42].

---

## 📌 Architecture & Features

- **Model Architecture:** ResNet50 Transfer Learning with Dropout & Batch Normalization[cite: 42].
- **Model Evaluation:** Automated Confusion Matrix and ROC Curve chart generation[cite: 42].
- **REST API Inference:** FastAPI backend serving predictions[cite: 42].
- **Web Interface:** Interactive Streamlit dashboard for scan uploads[cite: 42].
- **DevOps:** Dockerized deployment (`Dockerfile`) and GitHub Actions CI/CD automation (`ci-cd.yml`)[cite: 42].

---

## 🚀 How to Run

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
