# src/MNIST/TorchLightning/data_module_ocean.py

from __future__ import annotations

from typing import Optional, List

import lightning as L
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from src.shared.ocean_stress import prepare_ecological_stress_dataset


class OceanStressDataset(Dataset):
    """
    Simple tabular dataset for ecological stress classification.

    Features (default):
        - Latitude
        - Longitude
        - SST (°C)
        - pH Level
        - Species Observed

    Target:
        - EcologicalStressClass mapped to {Low:0, Medium:1, High:2}
    """

    def __init__(
        self,
        df,
        feature_cols: Optional[List[str]] = None,
        class_mapping: Optional[dict] = None,
    ):
        # Work on a copy so we don't mutate the original
        df = df.copy().reset_index(drop=True)

        if feature_cols is None:
            feature_cols = [
                "Latitude",
                "Longitude",
                "SST (°C)",
                "pH Level",
                "Species Observed",
            ]
        self.feature_cols = feature_cols

        if class_mapping is None:
            class_mapping = {"Low": 0, "Medium": 1, "High": 2}
        self.class_mapping = class_mapping

        # --- Ensure numeric features ---
        # Coerce to numeric; non‐numeric values become NaN
        df[self.feature_cols] = df[self.feature_cols].apply(
            pd.to_numeric, errors="coerce"
        )
        # Fill NaNs with column means (simple baseline strategy)
        df[self.feature_cols] = df[self.feature_cols].fillna(
            df[self.feature_cols].mean()
        )

        self.df = df
        self.targets = (
            self.df["EcologicalStressClass"].map(self.class_mapping).astype(int)
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        # Use to_numpy to guarantee a numeric array
        x = row[self.feature_cols].to_numpy(dtype="float32")
        x = torch.from_numpy(x)
        y = torch.tensor(self.targets.iloc[idx], dtype=torch.long)
        return x, y


class OceanEcologicalStressDataModule(L.LightningDataModule):
    """
    LightningDataModule for the ocean ecological stress dataset.

    - loads CSV via shared.ocean_stress helpers
    - creates train/val/test splits
    - returns DataLoaders for classification
    """

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 4,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        # Nothing to download; CSV is local and loaded in setup()
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage not in (None, "fit", "validate", "test"):
            return

        df = prepare_ecological_stress_dataset()

        # Drop rows without a class label, just in case
        df = df.dropna(subset=["EcologicalStressClass"])

        # Basic train/val/test split
        # First split off test set
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df["EcologicalStressClass"],
        )

        # Then split train/val
        val_relative_size = self.val_size / (1.0 - self.test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_relative_size,
            random_state=self.random_state,
            stratify=train_val_df["EcologicalStressClass"],
        )

        self.train_dataset = OceanStressDataset(train_df)
        self.val_dataset = OceanStressDataset(val_df)
        self.test_dataset = OceanStressDataset(test_df)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
