#!/usr/bin/env python3
"""
Download and prepare datasets for the headline_shift project.
"""

import os, sys, zipfile, glob
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import DATA_DIR, PROCESSED_DIR, TARGET_PUBLICATIONS, YEAR_RANGE, set_seed

set_seed(42)

KAGGLE_DATASET = "jordankrishnayah/45m-headlines-from-2007-2022-10-largest-sites"
HEADLINES_OUT  = os.path.join(PROCESSED_DIR, "headlines_filtered.csv")
QBIAS_RAW      = os.path.join(DATA_DIR, "qbias", "allsides_balanced_news_headlines-texts.csv")
QBIAS_OUT      = os.path.join(PROCESSED_DIR, "qbias_clean.csv")

os.makedirs(PROCESSED_DIR, exist_ok=True)


# ── 1. QBias ───────────────────────────────────────────────────────────────
def process_qbias():
    print("[1/3] Processing QBias data …")
    if not os.path.exists(QBIAS_RAW):
        print("  ⚠ QBias CSV not found. Run: git clone https://github.com/irgroup/Qbias data/qbias")
        return False
    df = pd.read_csv(QBIAS_RAW)
    df = df[["heading", "source", "bias_rating"]].rename(
        columns={"heading": "headline", "bias_rating": "label"}
    )
    df = df.dropna(subset=["headline", "label"])
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["left", "center", "right"])]
    df.to_csv(QBIAS_OUT, index=False)
    print(f"  ✓ Saved {len(df)} records → {QBIAS_OUT}")
    return True


# ── 2. Kaggle download ────────────────────────────────────────────────────
def find_zip() -> str | None:
    """Find the downloaded zip regardless of exact filename."""
    patterns = [
        os.path.join(DATA_DIR, "*.zip"),
        os.path.join(DATA_DIR, "**", "*.zip"),
    ]
    for p in patterns:
        matches = glob.glob(p, recursive=True)
        if matches:
            return matches[0]
    return None


def download_via_kagglehub() -> bool:
    """Try kagglehub — handles auth interactively."""
    try:
        import kagglehub
        print("  Trying kagglehub …")
        path = kagglehub.dataset_download(KAGGLE_DATASET)
        print(f"  ✓ kagglehub downloaded to: {path}")

        # copy CSVs into DATA_DIR so the rest of the pipeline finds them
        csv_files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
        if not csv_files:
            print("  ⚠ No CSVs found in kagglehub cache")
            return False

        for f in csv_files:
            dest = os.path.join(DATA_DIR, os.path.basename(f))
            if not os.path.exists(dest):
                import shutil
                shutil.copy(f, dest)
                print(f"  Copied {os.path.basename(f)} → {DATA_DIR}")
        return True

    except Exception as e:
        print(f"  ⚠ kagglehub failed: {e}")
        return False


def download_via_cli() -> bool:
    """Try kaggle CLI — requires ~/.kaggle/kaggle.json."""
    import subprocess
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json):
        print("  ⚠ ~/.kaggle/kaggle.json not found, skipping CLI download")
        return False

    print("  Trying kaggle CLI …")
    result = subprocess.run(
        ["kaggle", "datasets", "download", KAGGLE_DATASET, "-p", DATA_DIR, "--force"],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode == 0:
        zip_path = find_zip()
        if zip_path:
            print(f"  ✓ CLI downloaded: {zip_path}")
            return True
    print(f"  ⚠ CLI failed: {result.stderr[:300]}")
    return False


def extract_zip() -> bool:
    """Unzip whatever zip is in DATA_DIR."""
    zip_path = find_zip()
    if not zip_path:
        print("  ⚠ No zip found to extract")
        return False

    print(f"  Extracting {zip_path} …")
    with zipfile.ZipFile(zip_path) as zf:
        print(f"  Contents: {zf.namelist()}")
        zf.extractall(DATA_DIR)
    print("  ✓ Extracted")
    return True


# ── 3. Filter to headlines_filtered.csv ───────────────────────────────────
# Loose publication matching — prints what it finds so you can debug
PUB_MAP = {
    "cnn":              "CNN",
    "fox news":         "Fox News",
    "fox":              "Fox News",
    "washington post":  "Washington Post",
    "new york times":   "New York Times",
    "nyt":              "New York Times",
    "nytimes":          "New York Times",
}

def match_pub(val: str):
    val = str(val).lower().strip()
    for key, name in PUB_MAP.items():
        if key in val:
            return name
    return None


def extract_and_filter() -> bool:
    """Find CSVs in DATA_DIR, sniff columns, filter, save."""
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        print("  ⚠ No CSV files found in data/ after download")
        return False

    print(f"  Found CSVs: {[os.path.basename(f) for f in csv_files]}")
    frames = []

    for csv_path in csv_files:
        print(f"\n  Processing {os.path.basename(csv_path)} …")

        # sniff columns from first chunk
        sample = pd.read_csv(csv_path, nrows=5)
        print(f"  Columns detected: {sample.columns.tolist()}")
        print(f"  Sample row:\n{sample.iloc[0].to_dict()}")

        # map columns flexibly
        col_map = {}
        for c in sample.columns:
            cl = c.lower().strip()
            if cl in ["title", "headline", "heading"] and "headline" not in col_map:
                col_map["headline"] = c
            if cl in ["publication", "source", "publisher", "outlet"] and "publication" not in col_map:
                col_map["publication"] = c
            if cl in ["date", "publish_date", "published", "datetime"] and "date" not in col_map:
                col_map["date"] = c

        missing = [k for k in ["headline", "publication", "date"] if k not in col_map]
        if missing:
            print(f"  ⚠ Could not find columns for: {missing} — skipping this file")
            print(f"  Tip: check the column names above and update col_map manually if needed")
            continue

        print(f"  Column mapping: {col_map}")

        # read in chunks (file is huge)
        chunk_iter = pd.read_csv(
            csv_path,
            usecols=list(col_map.values()),
            chunksize=100_000,
            low_memory=False,
        )
        kept = 0
        for chunk in chunk_iter:
            chunk = chunk.rename(columns={v: k for k, v in col_map.items()})
            chunk["publication"] = chunk["publication"].apply(match_pub)
            chunk = chunk.dropna(subset=["publication"])
            chunk["date"] = pd.to_datetime(
                chunk["date"].astype(str).str.replace(r"\.0$", "", regex=True).str[:8],
                format="%Y%m%d",
                errors="coerce"
                    )
            chunk = chunk.dropna(subset=["date"])
            chunk = chunk[
                (chunk["date"].dt.year >= YEAR_RANGE[0]) &
                (chunk["date"].dt.year <= YEAR_RANGE[1])
            ]
            if len(chunk) > 0:
                frames.append(chunk[["publication", "date", "headline"]])
                kept += len(chunk)

        print(f"  ✓ Kept {kept} rows from this file")

    if not frames:
        print("  ❌ No matching rows found across all CSVs")
        print(f"  Publications being matched: {list(PUB_MAP.keys())}")
        print(f"  Year range: {YEAR_RANGE}")
        return False

    df = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    df.to_csv(HEADLINES_OUT, index=False)
    print(f"\n  ✓ Saved {len(df)} filtered headlines → {HEADLINES_OUT}")
    print(f"  Publication breakdown:\n{df['publication'].value_counts().to_string()}")
    return True


# ── main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Headline Shift – Data Download & Preparation")
    print("=" * 60)

    process_qbias()

    print("\n[2/3] Downloading Kaggle dataset …")

    if os.path.exists(HEADLINES_OUT):
        print(f"  Already processed at {HEADLINES_OUT}, skipping.")
    else:
        # check if CSVs are already in data/ (e.g. manually placed)
        existing_csvs = glob.glob(os.path.join(DATA_DIR, "*.csv"))
        already_downloaded = len(existing_csvs) > 0

        if not already_downloaded:
            success = download_via_kagglehub()
            # if kagglehub didn't give us CSVs, try CLI + unzip
            if not success or not glob.glob(os.path.join(DATA_DIR, "*.csv")):
                download_via_cli()
                extract_zip()

        if not extract_and_filter():
            print("\n  ❌ Could not build headlines_filtered.csv from real data.")
            print("  Check the debug output above to see what columns were found.")
            print("  Fix PUB_MAP or YEAR_RANGE in this script if needed, then re-run.")
            sys.exit(1)   # <-- hard fail instead of silent synthetic fallback

    print("\n[3/3] Data summary:")
    if os.path.exists(QBIAS_OUT):
        df = pd.read_csv(QBIAS_OUT)
        print(f"  QBias:     {len(df)} rows | Labels: {df['label'].value_counts().to_dict()}")
    if os.path.exists(HEADLINES_OUT):
        df = pd.read_csv(HEADLINES_OUT)
        print(f"  Headlines: {len(df)} rows | Pubs: {df['publication'].value_counts().to_dict()}")
    print("\n✓ Done.")


if __name__ == "__main__":
    main()
