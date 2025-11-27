# src/Keras/data_mnist.py

from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf

from src.config import MNISTConfig


def prepare_mnist_datasets(
    cfg: MNISTConfig,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Load MNIST and return train/val/test as tf.data.Datasets."""

    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalise to [0, 1]
    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Flatten to vectors [batch, 784] to match your Lightning MLP
    x_train_full = x_train_full.reshape(-1, cfg.input_dim)
    x_test = x_test.reshape(-1, cfg.input_dim)

    # Create validation split from training set
    n_total = x_train_full.shape[0]
    n_val = int(cfg.val_split * n_total)
    n_train = n_total - n_val

    x_train, x_val = x_train_full[:n_train], x_train_full[n_train:]
    y_train, y_val = y_train_full[:n_train], y_train_full[n_train:]

    # Build tf.data pipelines
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(buffer_size=n_train)
        .batch(cfg.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(cfg.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(cfg.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, test_ds
