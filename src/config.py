from dataclasses import dataclass


@dataclass
class MNISTConfig:
    # Data
    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 4
    val_split: float = 0.1

    # Model
    input_dim: int = 28 * 28
    hidden_dim: int = 512
    num_classes: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # Training
    max_epochs: int = 10
    log_dir: str = "logs"
    experiment_name: str = "mnist_pl"
