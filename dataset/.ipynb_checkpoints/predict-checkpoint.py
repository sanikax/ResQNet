"""
ResQNet - Ambulance Detection System
predict.py: Single-image prediction with confidence score and visualization

Author: [Your Name]
College: Savitribai Phule Pune University (SPPU)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from model import IMG_HEIGHT, IMG_WIDTH

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH  = "models/resqnet_best.h5"
IMG_SIZE    = (IMG_HEIGHT, IMG_WIDTH)

# Must match training class_indices — verify with train output
CLASS_NAMES = {0: "Ambulance 🚑", 1: "Non-Ambulance 🚗"}
THRESHOLD   = 0.5   # Confidence threshold for positive (ambulance) prediction


# ─────────────────────────────────────────────
# LOAD MODEL (once, at module level)
# ─────────────────────────────────────────────
print("[INFO] Loading ResQNet model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] Model ready.\n")


# ─────────────────────────────────────────────
# CORE PREDICTION FUNCTION
# ─────────────────────────────────────────────
def predict_image(img_path: str, show_plot: bool = True) -> dict:
    """
    Predict whether an image is an Ambulance or Non-Ambulance.

    Args:
        img_path   : Path to the image file (.jpg / .png / .jpeg)
        show_plot  : Display visualization with result overlay

    Returns:
        dict with keys: label, confidence, is_ambulance, raw_score
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    # ── Preprocess ───────────────────────────
    img = image.load_img(img_path, target_size=IMG_SIZE)   # Resize to 128×128
    img_array = image.img_to_array(img)                    # Convert to numpy
    img_array = img_array / 255.0                          # Normalize [0, 1]
    img_batch = np.expand_dims(img_array, axis=0)          # Add batch dim → (1,128,128,3)

    # ── Predict ──────────────────────────────
    raw_score    = float(model.predict(img_batch, verbose=0)[0][0])
    is_ambulance = raw_score < THRESHOLD  # Low score = class 0 = ambulance (verify with your class_indices)
    confidence   = (1 - raw_score) if is_ambulance else raw_score
    label        = CLASS_NAMES[0] if is_ambulance else CLASS_NAMES[1]

    result = {
        "label"        : label,
        "confidence"   : round(confidence * 100, 2),
        "is_ambulance" : is_ambulance,
        "raw_score"    : round(raw_score, 4)
    }

    # ── Print Result ─────────────────────────
    print("─" * 45)
    print(f"  Image      : {os.path.basename(img_path)}")
    print(f"  Prediction : {label}")
    print(f"  Confidence : {result['confidence']}%")
    print(f"  Raw Score  : {result['raw_score']}")
    print("─" * 45)

    # ── Visualize ────────────────────────────
    if show_plot:
        _visualize_prediction(img, result, img_path)

    return result


def _visualize_prediction(img, result: dict, img_path: str):
    """Show the image with prediction overlay."""
    color = "#27ae60" if result["is_ambulance"] else "#e74c3c"
    label = result["label"]
    conf  = result["confidence"]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(
        f"{label}\nConfidence: {conf}%",
        fontsize=13, fontweight='bold', color=color, pad=12
    )

    # Colored border to indicate prediction
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(4)

    plt.tight_layout()
    plt.savefig("outputs/last_prediction.png", dpi=150, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────
# BATCH PREDICTION
# ─────────────────────────────────────────────
def predict_batch(image_paths: list) -> list:
    """
    Run predictions on a list of image paths.

    Returns:
        list of result dicts
    """
    results = []
    for path in image_paths:
        try:
            result = predict_image(path, show_plot=False)
            results.append({"path": path, **result})
        except Exception as e:
            print(f"[ERROR] Skipping {path}: {e}")
    return results


# ─────────────────────────────────────────────
# CLI USAGE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py test_image.jpg")
        sys.exit(1)

    img_path = sys.argv[1]
    predict_image(img_path, show_plot=True)
