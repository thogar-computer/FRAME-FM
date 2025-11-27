# src/Keras/data_ocean.py

from __future__ import annotations

from typing import Tuple, List

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.config import OceanConfig
from src.shared.ocean_stress import prepare_ecological_stress_dataset


DEFAULT_FEATURE_COLS: List[str] = [
    "Latitude",
    "Longitude",
    "SST (°C)",
    "pH Level",
    "Species Observed",
]

CLASS_MAPPING = {"Low": 0, "Medium": 1, "High": 2}


def _prepare_features_and_labels(
    cfg: OceanConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = prepare_ecological_stress_dataset(cfg.filename)

    # Drop rows without a class (just in case)
    df = df.dropna(subset=["EcologicalStressClass"]).reset_index(drop=True)

    # Ensure numeric features
    df[DEFAULT_FEATURE_COLS] = df[DEFAULT_FEATURE_COLS].apply(
        pd.to_numeric, errors="coerce"
    )
    df[DEFAULT_FEATURE_COLS] = df[DEFAULT_FEATURE_COLS].fillna(
        df[DEFAULT_FEATURE_COLS].mean()
    )

    X = df[DEFAULT_FEATURE_COLS].to_numpy(dtype="float32")
    y = df["EcologicalStressClass"].map(CLASS_MAPPING).to_numpy(dtype="int64")

    # First split off test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    # Then split train/val
    val_relative_size = cfg.val_size / (1.0 - cfg.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_relative_size,
        random_state=cfg.random_state,
        stratify=y_train_val,
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def prepare_ocean_datasets(
    cfg: OceanConfig,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Prepare train/val/test tf.data.Datasets for the ocean stress classification."""

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = _prepare_features_and_labels(cfg)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(buffer_size=len(X_train))
        .batch(cfg.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(cfg.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test))
        .batch(cfg.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, test_ds
