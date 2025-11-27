# src/Keras/train.py

from __future__ import annotations

import argparse

import tensorflow as tf

from lightning.pytorch.loggers import (
    MLFlowLogger,
)  # optional: if you want MLflow here too

from src.config import MNISTConfig, OceanConfig
from .data_module_mnist import prepare_mnist_datasets
from .model import build_mnist_model
from .data_module_ocean import prepare_ocean_datasets
from .model_ocean import build_ocean_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Keras models")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mnist", action="store_true", help="Train Keras MNIST model")
    group.add_argument(
        "--ocean", action="store_true", help="Train Keras ocean stress model"
    )

    parser.add_argument("--epochs", type=int, default=None, help="Override max_epochs")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override batch size"
    )

    # Optional: MLflow toggle
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Use MLflow autologging for Keras runs",
    )

    return parser.parse_args()


def train_mnist(args: argparse.Namespace) -> None:
    cfg = MNISTConfig()
    if args.epochs is not None:
        cfg.max_epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    train_ds, val_ds, test_ds = prepare_mnist_datasets(cfg)
    model = build_mnist_model(cfg)

    if args.mlflow:
        import mlflow
        import mlflow.keras

        mlflow.set_experiment(cfg.experiment_name + "_keras")
        mlflow.keras.autolog()
        with mlflow.start_run():
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=cfg.max_epochs,
                verbose=1,
            )
            test_loss, test_acc = model.evaluate(test_ds, verbose=0)
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    else:
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg.max_epochs,
            verbose=1,
        )
        test_loss, test_acc = model.evaluate(test_ds, verbose=0)
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")


def train_ocean(args: argparse.Namespace) -> None:
    cfg = OceanConfig()
    if args.epochs is not None:
        cfg.max_epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    train_ds, val_ds, test_ds = prepare_ocean_datasets(cfg)
    model = build_ocean_model(cfg)

    if args.mlflow:
        import mlflow
        import mlflow.keras

        mlflow.set_experiment(cfg.experiment_name + "_keras")
        mlflow.keras.autolog()
        with mlflow.start_run():
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=cfg.max_epochs,
                verbose=1,
            )
            test_loss, test_acc = model.evaluate(test_ds, verbose=0)
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    else:
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg.max_epochs,
            verbose=1,
        )
        test_loss, test_acc = model.evaluate(test_ds, verbose=0)
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")


def main() -> None:
    args = parse_args()

    # Optional: deterministic behaviour
    tf.random.set_seed(42)

    if args.mnist:
        train_mnist(args)
    elif args.ocean:
        train_ocean(args)


if __name__ == "__main__":
    main()
