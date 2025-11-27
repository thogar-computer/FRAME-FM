# mnist/train.py

import argparse
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger

from .data_module_mnist import MNISTDataModule
from .model import MNISTClassifier
from ..config import MNISTConfig


def train(options: argparse.Namespace) -> None:
    """
    Select dataset/model based on CLI options and train with a shared
    logger, callbacks, and Lightning Trainer.
    """

    if options.mnist:
        cfg = MNISTConfig()

        # Apply CLI overrides
        if options.epochs is not None:
            cfg.max_epochs = options.epochs
        if options.batch_size is not None:
            cfg.batch_size = options.batch_size

        dm = MNISTDataModule(
            data_dir=cfg.data_dir,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            val_split=cfg.val_split,
        )

        model = MNISTClassifier(
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            num_classes=cfg.num_classes,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        experiment_name = cfg.experiment_name
        ckpt_prefix = "mnist"
        max_epochs = cfg.max_epochs

    elif options.ocean:
        # When you're ready, uncomment + implement these:
        #
        # from .data_module_ocean import OceanEcologicalStressDataModule
        # from .model_ocean import OceanStressClassifier
        #
        # batch_size = options.batch_size or 64
        # max_epochs = options.epochs or 20
        #
        # dm = OceanEcologicalStressDataModule(batch_size=batch_size)
        # model = OceanStressClassifier()
        #
        # experiment_name = "ocean_ecological_stress"
        # ckpt_prefix = "ocean"
        #
        raise NotImplementedError("Ocean training not yet implemented.")
    else:
        raise ValueError("No dataset option provided (use --mnist or --ocean).")

    # MLflow logger (shared)
    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri="file:./mlruns",  # or your MLflow server URI
    )

    # Shared checkpointing behaviour
    checkpoint_cb = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename=f"{ckpt_prefix}" + "-{epoch:02d}-{val_acc:.4f}",
    )

    # Shared early stopping behaviour
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=5,
        verbose=True,
    )

    # Shared Trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        logger=mlf_logger,
        callbacks=[checkpoint_cb, early_stop_cb],
        log_every_n_steps=50,
    )

    trainer.fit(model, dm)
    trainer.test(model=model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML models in TorchLightning")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mnist", action="store_true", help="Train on MNIST dataset")
    group.add_argument(
        "--ocean", action="store_true", help="Train on ocean ecological stress dataset"
    )

    # Optional overrides
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override max_epochs from config"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override batch size from config"
    )

    args = parser.parse_args()
    train(options=args)
