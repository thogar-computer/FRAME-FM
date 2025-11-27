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


@dataclass
class OceanConfig:
    # Data
    data_dir: str = "./data"
    filename: str = "realistic_ocean_climate_dataset.csv"
    batch_size: int = 64
    num_workers: int = 4
    val_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42

    # Model
    input_dim: int = 5  # Latitude, Longitude, SST, pH, Species Observed
    hidden_dim: int = 256
    num_classes: int = 3  # Low, Medium, High
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # Training
    max_epochs: int = 20
    experiment_name: str = "ocean_ecological_stress_pl"
