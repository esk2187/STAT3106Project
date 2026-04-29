#!/usr/bin/env python3
"""
Main pipeline script for The Headline Shift project.
Runs everything end-to-end: data prep → training → inference → visualization.

Usage:
    python run_pipeline.py                                     # dual-head DistilBERT (primary)
    python run_pipeline.py --model ablation                    # DistilBERT ideology-only
    python run_pipeline.py --model roberta                     # RoBERTa single-task (Baseline 2)
    python run_pipeline.py --model roberta_multitask \\
        --emotionality-labels data/processed/emotionality_labels.csv
    python run_pipeline.py --skip-transformer                  # TF-IDF baseline only
    python run_pipeline.py --data-only                         # just download/prep data
    python run_pipeline.py --model ablation --no-inference     # train only, skip scoring
    python run_pipeline.py --model multitask \\
        --emotionality-labels data/processed/emotionality_labels.csv
"""

import argparse
import os
import sys
import time

# make sure we can import src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import set_seed, PROCESSED_DIR, RESULTS_DIR
set_seed()


def main():
    parser = argparse.ArgumentParser(description="Headline Shift – full pipeline")
    parser.add_argument(
        "--model",
        choices=["multitask", "ablation", "roberta", "roberta_multitask"],
        default="multitask",
        help=(
            "Which transformer to train: "
            "'multitask' = dual-head DistilBERT (primary model), "
            "'ablation' = DistilBERT ideology-only, "
            "'roberta' = RoBERTa-base single-task (Baseline 2), "
            "'roberta_multitask' = dual-head RoBERTa (extended comparison)"
        ),
    )
    parser.add_argument(
        "--emotionality-labels",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to emotionality labels CSV exported from the active learning app "
             "(required to use the emotionality head in --model multitask).",
    )
    parser.add_argument("--skip-transformer", action="store_true",
                        help="Skip all transformer training (run TF-IDF baseline only)")
    parser.add_argument("--no-inference", action="store_true",
                        help="Skip inference and visualization (training only — use to compare models faster)")
    parser.add_argument("--data-only", action="store_true",
                        help="Only run data download/preparation")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of transformer fine-tuning epochs (default 1 for speed; use 3-5 for final run)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for transformer training (default 64)")
    args = parser.parse_args()

    start = time.time()
    print("=" * 60)
    print("  The Headline Shift: Full Pipeline")
    print("=" * 60)

    # ── Step 1: Data ──────────────────────────────────────────────────────
    print("\n▶ Step 1: Data Preparation")
    from data.download_data import main as download_main
    download_main()

    if args.data_only:
        print("\n✓ Data preparation complete. Exiting (--data-only).")
        return

    # ── Step 2: Load & Split ──────────────────────────────────────────────
    print("\n▶ Step 2: Loading & Splitting QBias Data")
    from src.data_loader import load_qbias, split_qbias, load_headlines

    qbias = load_qbias()
    print(f"  QBias: {len(qbias)} samples")
    print(f"  Label distribution: {qbias['label'].value_counts().to_dict()}")

    train_df, val_df, test_df = split_qbias(qbias)
    print(f"  Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # ── Step 3: Baseline Model ────────────────────────────────────────────
    print("\n▶ Step 3: Training Baseline Model (TF-IDF + Logistic Regression)")
    from src.baseline_model import train_baseline
    baseline_clf, baseline_metrics = train_baseline(train_df, val_df, test_df)

    # ── Step 4: Transformer Model ─────────────────────────────────────────
    transformer_clf        = None
    multitask_clf          = None
    roberta_clf            = None
    roberta_multitask_clf  = None

    if args.skip_transformer:
        print("\n▶ Step 4: Skipping transformer (--skip-transformer)")

    elif args.model == "roberta":
        print("\n▶ Step 4: Training Baseline 2 — RoBERTa-base (single-task)")
        from src.transformer_model import train_roberta
        roberta_clf, _ = train_roberta(
            train_df, val_df, test_df,
            epochs=args.epochs, batch_size=args.batch_size,
        )

    elif args.model == "roberta_multitask":
        print("\n▶ Step 4: Training RoBERTa Dual-Head (ideology + emotionality)")
        emotion_df = None
        if args.emotionality_labels:
            import pandas as pd
            if os.path.exists(args.emotionality_labels):
                emotion_df = pd.read_csv(args.emotionality_labels)
                print(f"  Loaded {len(emotion_df)} emotionality labels from {args.emotionality_labels}")
            else:
                print(f"  ⚠  --emotionality-labels path not found: {args.emotionality_labels}")
        from src.transformer_model import train_roberta_multitask
        roberta_multitask_clf, _ = train_roberta_multitask(
            train_df, val_df, test_df,
            emotion_df=emotion_df,
            epochs=args.epochs, batch_size=args.batch_size,
        )

    elif args.model == "ablation":
        print("\n▶ Step 4: Training Ablation — DistilBERT ideology-only")
        from src.transformer_model import train_multitask
        multitask_clf, _ = train_multitask(
            train_df, val_df, test_df,
            emotion_df=None,
            epochs=args.epochs, batch_size=args.batch_size,
        )

    else:  # multitask (default)
        print("\n▶ Step 4: Training Primary Model — Multi-task DistilBERT")
        emotion_df = None
        if args.emotionality_labels:
            import pandas as pd
            if os.path.exists(args.emotionality_labels):
                emotion_df = pd.read_csv(args.emotionality_labels)
                print(f"  Loaded {len(emotion_df)} emotionality labels from {args.emotionality_labels}")
            else:
                print(f"  ⚠  --emotionality-labels path not found: {args.emotionality_labels}")
        else:
            print("  ℹ  No --emotionality-labels provided. "
                  "Run the active learning app and export labels to enable multi-task training.")

        from src.transformer_model import train_multitask
        multitask_clf, _ = train_multitask(
            train_df, val_df, test_df,
            emotion_df=emotion_df,
            epochs=args.epochs, batch_size=args.batch_size,
        )

    # ── Step 5: Inference ─────────────────────────────────────────────────
    if args.no_inference:
        print("\n▶ Steps 5-6: Skipping inference and visualization (--no-inference)")
        print("  Run without --no-inference when ready to score headlines with the best model.")
    else:
        print("\n▶ Step 5: Running Inference Pipeline")
        headlines = load_headlines()
        print(f"  Loaded {len(headlines)} headlines")

        from src.inference import run_inference
        scored_df = run_inference(
            headlines,
            baseline_model=baseline_clf,
            transformer_model=transformer_clf,
            multitask_model=multitask_clf,
            roberta_model=roberta_clf,
            roberta_multitask_model=roberta_multitask_clf,
        )

        # ── Step 6: Visualizations ────────────────────────────────────────────
        print("\n▶ Step 6: Generating Visualizations")
        from src.time_series import generate_all_plots
        plot_paths = generate_all_plots(scored_df)

    # ── Done ──────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"  ✓ Pipeline complete in {elapsed:.1f}s")
    print(f"  Results: {RESULTS_DIR}")
    print(f"  Plots:   {os.path.join(RESULTS_DIR, '..', 'plots')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
