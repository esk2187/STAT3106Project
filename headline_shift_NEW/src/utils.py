"""
Shared utilities for the headline_shift project.
Constants, helpers, and common config live here.
"""

import os
import random
import numpy as np

# ── project paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
EMOTIONALITY_LABELS_PATH = os.path.join(PROCESSED_DIR, "emotionality_labels.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
RESULTS_DIR = os.path.join(OUTPUTS_DIR, "results")

# make sure output dirs exist
for d in [PROCESSED_DIR, MODELS_DIR, PLOTS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── constants ──────────────────────────────────────────────────────────────
TARGET_PUBLICATIONS = ["CNN", "Fox News", "Washington Post", "New York Times"]
IDEOLOGY_LABELS = ["left", "center", "right"]
IDEOLOGY_LABEL2ID = {l: i for i, l in enumerate(IDEOLOGY_LABELS)}
IDEOLOGY_ID2LABEL = {i: l for i, l in enumerate(IDEOLOGY_LABELS)}
ELECTION_YEARS = [2014, 2016, 2018, 2020, 2022]  # midterms + presidential
YEAR_RANGE = (2013, 2022)

# ── reproducibility ───────────────────────────────────────────────────────
SEED = 42

def set_seed(seed: int = SEED):
    """Lock down randomness for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def label_to_numeric(label: str) -> float:
    """Map ideology labels to a -1/0/+1 numeric scale for trend analysis."""
    mapping = {"left": -1.0, "center": 0.0, "right": 1.0}
    return mapping.get(label, 0.0)
