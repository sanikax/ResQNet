"""
ResQNet - Ambulance Detection System
evaluate.py: Model evaluation — confusion matrix, classification report, ROC curve

Author: [Your Name]
College: Savitribai Phule Pune University (SPPU)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, ConfusionMatrixDisplay
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import IMG_HEIGHT, IMG_WIDTH

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH = "models/resqnet_best.h5"
TEST_DIR   = "dataset/test"
IMG_SIZE   = (IMG_HEIGHT, IMG_WIDTH)
BATCH_SIZE = 32
os.makedirs("outputs", exist_ok=True)


# ─────────────────────────────────────────────
# LOAD MODEL & TEST DATA
# ─────────────────────────────────────────────
print("[INFO] Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Class labels — Keras sorts folders alphabetically
class_names = list(test_generator.class_indices.keys())
print(f"Classes: {class_names}")


# ─────────────────────────────────────────────
# GENERATE PREDICTIONS
# ─────────────────────────────────────────────
print("[INFO] Running predictions on test set...")
y_pred_prob = model.predict(test_generator, verbose=1)  # Raw sigmoid scores
y_pred      = (y_pred_prob > 0.5).astype(int).flatten()  # Threshold at 0.5
y_true      = test_generator.classes                     # Ground-truth labels


# ─────────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=False, cmap='Blues')

    ax.set_title("ResQNet — Confusion Matrix", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150)
    plt.show()
    print("[INFO] Confusion matrix saved → outputs/confusion_matrix.png")

    # Manual breakdown
    tn, fp, fn, tp = cm.ravel()
    print(f"\nTrue Negatives  (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives  (TP): {tp}")


# ─────────────────────────────────────────────
# CLASSIFICATION REPORT
# ─────────────────────────────────────────────
def print_classification_report(y_true, y_pred, class_names):
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\n── Classification Report ─────────────────")
    print(report)

    # Save to file
    with open("outputs/classification_report.txt", "w") as f:
        f.write("ResQNet - Classification Report\n")
        f.write("=" * 45 + "\n")
        f.write(report)
    print("[INFO] Report saved → outputs/classification_report.txt")


# ─────────────────────────────────────────────
# ROC CURVE
# ─────────────────────────────────────────────
def plot_roc_curve(y_true, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc     = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--',
             label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ResQNet — ROC Curve', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/roc_curve.png", dpi=150)
    plt.show()
    print(f"[INFO] ROC curve saved → outputs/roc_curve.png  |  AUC = {roc_auc:.4f}")


# ─────────────────────────────────────────────
# OVERALL TEST ACCURACY
# ─────────────────────────────────────────────
def evaluate_model(model, test_generator):
    loss, accuracy, precision, recall, auc_score = model.evaluate(
        test_generator, verbose=1
    )
    print("\n── Final Test Metrics ────────────────────")
    print(f"  Loss      : {loss:.4f}")
    print(f"  Accuracy  : {accuracy*100:.2f}%")
    print(f"  Precision : {precision*100:.2f}%")
    print(f"  Recall    : {recall*100:.2f}%")
    print(f"  AUC       : {auc_score:.4f}")


# ─────────────────────────────────────────────
# RUN ALL EVALUATIONS
# ─────────────────────────────────────────────
if __name__ == "__main__":
    evaluate_model(model, test_generator)
    plot_confusion_matrix(y_true, y_pred, class_names)
    print_classification_report(y_true, y_pred, class_names)
    plot_roc_curve(y_true, y_pred_prob)
