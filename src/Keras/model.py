# src/Keras/model_mnist.py

from __future__ import annotations

import tensorflow as tf
from src.config import MNISTConfig


def build_mnist_model(cfg: MNISTConfig) -> tf.keras.Model:
    """Build a simple MLP classifier for MNIST."""

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(cfg.input_dim,)),
            tf.keras.layers.Dense(cfg.hidden_dim, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(cfg.hidden_dim, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(cfg.num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=cfg.lr, weight_decay=cfg.weight_decay
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
