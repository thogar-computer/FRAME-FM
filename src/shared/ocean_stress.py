# src/shared/ocean_stress.py

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np


DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_climate_csv(
    filename: str = "realistic_ocean_climate_dataset.csv",
) -> pd.DataFrame:
    """Load the climate CSV from the project-level /data directory."""
    path = DATA_DIR / filename
    df = pd.read_csv(path)
    return df


def add_bleaching_score(df: pd.DataFrame) -> pd.DataFrame:
    """Map categorical Bleaching Severity to a numeric score."""
    mapping = {
        "None": 0,
        "Low": 1,
        "Medium": 2,
        "High": 3,
    }
    df = df.copy()
    df["BleachingScore"] = df["Bleaching Severity"].map(mapping).fillna(0).astype(int)
    return df


def add_sst_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Add an SST risk score based on temperature thresholds."""
    df = df.copy()
    sst = df["SST (°C)"]

    conditions = [
        sst < 26.0,
        (sst >= 26.0) & (sst < 28.0),
        (sst >= 28.0) & (sst < 30.0),
        sst >= 30.0,
    ]
    values = [0, 1, 2, 3]

    df["SSTRisk"] = np.select(conditions, values, default=0).astype(int)
    return df


def add_ph_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Add an acidification risk score based on pH."""
    df = df.copy()
    ph = df["pH Level"]

    conditions = [
        ph >= 8.0,
        (ph >= 7.9) & (ph < 8.0),
        (ph >= 7.8) & (ph < 7.9),
        ph < 7.8,
    ]
    values = [0, 1, 2, 3]

    df["PHRisk"] = np.select(conditions, values, default=0).astype(int)
    return df


def add_ecological_stress(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add combined ecological stress indicators:

    - BleachingScore (0–3)
    - SSTRisk (0–3)
    - PHRisk (0–3)
    - EcologicalStressScore (0–9)
    - EcologicalStressClass ("Low", "Medium", "High")
    """
    df = add_bleaching_score(df)
    df = add_sst_risk(df)
    df = add_ph_risk(df)

    df = df.copy()
    df["EcologicalStressScore"] = (
        df["BleachingScore"] + df["SSTRisk"] + df["PHRisk"]
    ).astype(int)

    # Classify into Low / Medium / High
    bins = [-1, 2, 5, 9]
    labels = ["Low", "Medium", "High"]
    df["EcologicalStressClass"] = pd.cut(
        df["EcologicalStressScore"], bins=bins, labels=labels
    )

    return df


def prepare_ecological_stress_dataset(
    filename: str = "realistic_ocean_climate_dataset.csv",
) -> pd.DataFrame:
    """
    Convenience function:
    - load CSV
    - add ecological stress columns
    """
    df = load_climate_csv(filename)
    df = add_ecological_stress(df)
    return df
