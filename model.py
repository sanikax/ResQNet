"""
ResQNet - Ambulance Detection System
model.py: CNN model architecture definition

Author: [Your Name]
College: Savitribai Phule Pune University (SPPU)
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization, Input
)
from tensorflow.keras.regularizers import l2

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
IMG_HEIGHT  = 128
IMG_WIDTH   = 128
NUM_CLASSES = 1          # Binary classification → sigmoid output


# ─────────────────────────────────────────────
# CUSTOM CNN MODEL
# ─────────────────────────────────────────────
def build_resqnet_cnn(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)):
    """
    ResQNet custom CNN architecture.

    Block structure:
        Block 1: 32 filters  – captures low-level edges, textures
        Block 2: 64 filters  – captures mid-level shapes, patterns
        Block 3: 128 filters – captures high-level semantic features
        Head   : Dense layers → binary classification

    Each conv block uses:
        Conv2D → BatchNorm → ReLU → MaxPooling → Dropout
    """
    model = Sequential(name="ResQNet_CNN")

    model.add(Input(shape=input_shape))

    # ── Block 1 ──────────────────────────────
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # ── Block 2 ──────────────────────────────
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # ── Block 3 ──────────────────────────────
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.40))

    # ── Classification Head ───────────────────
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.50))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.30))
    model.add(Dense(NUM_CLASSES, activation='sigmoid'))  # Binary output

    return model


# ─────────────────────────────────────────────
# TRANSFER LEARNING MODEL (MobileNetV2)
# ─────────────────────────────────────────────
def build_mobilenet_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)):
    """
    Transfer learning using MobileNetV2 pretrained on ImageNet.
    Recommended when dataset size is small (<5000 images per class).

    Strategy: Freeze base layers → train custom head → optionally fine-tune.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs  = tf.keras.Input(shape=input_shape)
    x       = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x       = base_model(x, training=False)
    x       = tf.keras.layers.GlobalAveragePooling2D()(x)
    x       = tf.keras.layers.Dense(128, activation='relu')(x)
    x       = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs, name="ResQNet_MobileNetV2")
    return model, base_model


if __name__ == "__main__":
    model = build_resqnet_cnn()
    model.summary()
