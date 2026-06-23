"""
ResQNet - Ambulance Detection System
train.py: Data preprocessing, augmentation, and model training

Author: [Your Name]
College: Savitribai Phule Pune University (SPPU)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint,
    ReduceLROnPlateau, CSVLogger
)
from model import build_resqnet_cnn, IMG_HEIGHT, IMG_WIDTH

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATASET_DIR  = "dataset"
TRAIN_DIR    = os.path.join(DATASET_DIR, "train")
TEST_DIR     = os.path.join(DATASET_DIR, "test")
MODEL_SAVE   = "models/resqnet_best.h5"
LOG_FILE     = "logs/training_log.csv"

BATCH_SIZE   = 32
EPOCHS       = 30
IMG_SIZE     = (IMG_HEIGHT, IMG_WIDTH)
LEARNING_RATE = 1e-3


# ─────────────────────────────────────────────
# STEP 1: DATA AUGMENTATION & PREPROCESSING
# ─────────────────────────────────────────────
# Training generator — applies augmentation to prevent overfitting
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,           # Normalize pixel values to [0, 1]
    rotation_range=20,           # Randomly rotate images ±20°
    width_shift_range=0.15,      # Horizontally shift up to 15%
    height_shift_range=0.15,     # Vertically shift up to 15%
    shear_range=0.10,            # Shear transformation
    zoom_range=0.20,             # Random zoom in/out
    horizontal_flip=True,        # Mirror images horizontally
    brightness_range=[0.8, 1.2], # Vary brightness
    fill_mode='nearest'          # Fill empty pixels with nearest value
)

# Validation/Test generator — only normalize, NO augmentation
val_datagen = ImageDataGenerator(rescale=1.0 / 255)


# ─────────────────────────────────────────────
# STEP 2: DATA LOADERS
# ─────────────────────────────────────────────
print("[INFO] Loading training data...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',         # Binary: ambulance=1, no_ambulance=0
    shuffle=True,
    seed=42
)

print("[INFO] Loading test/validation data...")
test_generator = val_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False                # Keep order for confusion matrix
)

# Display class mapping
print(f"\nClass indices: {train_generator.class_indices}")
# Expected: {'ambulance': 0, 'no_ambulance': 1}  (or reversed based on folder sort)


# ─────────────────────────────────────────────
# STEP 3: BUILD MODEL
# ─────────────────────────────────────────────
print("\n[INFO] Building ResQNet CNN model...")
model = build_resqnet_cnn(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Compile with Adam optimizer and binary crossentropy loss
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',  # Standard loss for binary classification
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)
model.summary()


# ─────────────────────────────────────────────
# STEP 4: CALLBACKS
# ─────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
os.makedirs("logs",   exist_ok=True)

callbacks = [
    # Save best model based on validation accuracy
    ModelCheckpoint(
        MODEL_SAVE,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    # Stop training if val_loss doesn't improve for 7 consecutive epochs
    EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce learning rate when training plateaus
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    # Save epoch-by-epoch metrics to CSV
    CSVLogger(LOG_FILE, append=False)
]


# ─────────────────────────────────────────────
# STEP 5: TRAIN MODEL
# ─────────────────────────────────────────────
print("\n[INFO] Starting training...\n")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=callbacks,
    verbose=1
)

print(f"\n[INFO] Best model saved to → {MODEL_SAVE}")


# ─────────────────────────────────────────────
# STEP 6: PLOT TRAINING HISTORY
# ─────────────────────────────────────────────
def plot_training_history(history):
    """Plot accuracy and loss curves for training vs validation."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ResQNet - Training History", fontsize=14, fontweight='bold')

    # — Accuracy Plot —
    axes[0].plot(history.history['accuracy'],     label='Train Accuracy', color='royalblue')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy',   color='darkorange')
    axes[0].set_title("Accuracy over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # — Loss Plot —
    axes[1].plot(history.history['loss'],     label='Train Loss', color='royalblue')
    axes[1].plot(history.history['val_loss'], label='Val Loss',   color='darkorange')
    axes[1].set_title("Loss over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/training_history.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("[INFO] Training history saved → outputs/training_history.png")


os.makedirs("outputs", exist_ok=True)
plot_training_history(history)
