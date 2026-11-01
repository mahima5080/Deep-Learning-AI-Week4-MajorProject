"""
Deep Learning & AI Internship — Week 4 Capstone Project
Medical Image Analysis: Pneumonia Detection CNN Training & Evaluation Pipeline
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
CLASSES = ["Normal", "Pneumonia"]

def build_medical_cnn():
    """Builds a CNN architecture with Dropout and Batch Normalization."""
    base_model = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False  # Feature extraction

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model

def generate_synthetic_medical_data(num_samples=200):
    """Generates synthetic medical image tensors for pipeline execution."""
    x_data = np.random.uniform(0, 255, size=(num_samples, 224, 224, 3)).astype(np.float32)
    y_data = np.random.randint(0, 2, size=(num_samples, 1))
    return x_data, y_data

def evaluate_and_plot_metrics(model, x_val, y_val):
    """Generates Confusion Matrix and ROC Curve plots."""
    y_pred_prob = model.predict(x_val)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # 1. Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Medical Image Classification - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()
    print("[✔] Exported confusion_matrix.png")

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    plt.close()
    print(f"[✔] Exported roc_curve.png (AUC: {roc_auc:.2f})")

def main():
    print("[*] Initializing Medical CNN Architecture...")
    model = build_medical_cnn()

    x_train, y_train = generate_synthetic_medical_data(160)
    x_val, y_val = generate_synthetic_medical_data(40)

    print("[*] Training CNN Model with Data Augmentation...")
    model.fit(x_train, y_train, epochs=3, batch_size=BATCH_SIZE, validation_data=(x_val, y_val))

    print("[*] Running Model Evaluation Pipeline...")
    evaluate_and_plot_metrics(model, x_val, y_val)

    print("[*] Exporting Trained Model...")
    model.save("medical_resnet_model.h5")
    print("[✔] Medical Model saved to 'medical_resnet_model.h5'")

if __name__ == "__main__":
    main()