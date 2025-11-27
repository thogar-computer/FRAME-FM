# mnist/train.py

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger


from .data_module import MNISTDataModule
from .model import MNISTClassifier
from ...config import MNISTConfig


def train():
    cfg = MNISTConfig()

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

    # --- MLflow logger instead of TensorBoard ---
    mlf_logger = MLFlowLogger(
        experiment_name=cfg.experiment_name,
        tracking_uri="file:./mlruns",  # or your MLflow server URI
    )

    checkpoint_cb = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="mnist-{epoch:02d}-{val_acc:.4f}",
    )

    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=5,
        verbose=True,
    )

    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        devices="auto",
        logger=mlf_logger,
        callbacks=[checkpoint_cb, early_stop_cb],
        log_every_n_steps=50,
    )

    trainer.fit(model, dm)
    trainer.test(model=model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    train()
