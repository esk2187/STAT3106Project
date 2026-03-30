"""
Data loading and preprocessing for headline_shift.
Handles both QBias (labeled ideology) and the large headline corpus.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import (
    PROCESSED_DIR, IDEOLOGY_LABELS, IDEOLOGY_LABEL2ID, SEED, set_seed
)

set_seed()

# ── paths ──────────────────────────────────────────────────────────────────
QBIAS_PATH = os.path.join(PROCESSED_DIR, "qbias_clean.csv")
HEADLINES_PATH = os.path.join(PROCESSED_DIR, "headlines_filtered.csv")


def load_qbias(path: str = QBIAS_PATH) -> pd.DataFrame:
    """Load cleaned QBias data with headline + label columns."""
    df = pd.read_csv(path)
    df["label_id"] = df["label"].map(IDEOLOGY_LABEL2ID)
    return df


def split_qbias(df: pd.DataFrame, test_size=0.15, val_size=0.15):
    """
    Three-way stratified split: train / val / test.
    Returns (train_df, val_df, test_df).
    """
    # first: separate test
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=SEED
    )
    # then: split remaining into train + val
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val, stratify=train_val["label"], random_state=SEED
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def load_headlines(path: str = HEADLINES_PATH) -> pd.DataFrame:
    """Load the large headline corpus (real or synthetic)."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["headline"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)
    return df


def get_sample(df: pd.DataFrame, n: int = 1000, per_pub: bool = True) -> pd.DataFrame:
    """Grab a random sample, optionally balanced per publication."""
    if per_pub:
        return df.groupby("publication").apply(
            lambda x: x.sample(min(n, len(x)), random_state=SEED)
        ).reset_index(drop=True)
    return df.sample(min(n, len(df)), random_state=SEED).reset_index(drop=True)
