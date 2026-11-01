"""
Deep Learning & AI Internship — Week 4 Capstone
FastAPI Inference Server for Chest X-Ray Medical Analysis
"""

import io
import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Medical Chest X-Ray Diagnosis API",
    description="Deep Learning API for Automated Pneumonia Detection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "medical_resnet_model.h5"
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("[✔] Medical ResNet Model loaded successfully.")
    except Exception as e:
        print(f"[!] Warning: Model file not found. Run train_eval.py first.")

def preprocess_xray(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

@app.get("/health")
def health_check():
    return {"status": "Healthy", "service": "Medical Image Analysis API"}

@app.post("/predict")
async def predict_xray(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model missing. Run train_eval.py first.")

    contents = await file.read()
    tensor = preprocess_xray(contents)
    prediction_score = float(model.predict(tensor)[0][0])

    diagnosis = "Pneumonia Detected" if prediction_score > 0.5 else "Normal (Healthy)"
    confidence = prediction_score if prediction_score > 0.5 else (1 - prediction_score)

    return {
        "filename": file.filename,
        "diagnosis": diagnosis,
        "confidence": f"{confidence * 100:.2f}%",
        "raw_score": round(prediction_score, 4)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)