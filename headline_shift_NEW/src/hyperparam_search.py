"""
src/hyperparam_search.py

Hyperparameter search for the dual-head DistilBERT model.
Searches over learning rates and emotionality loss weights using the
validation set — never touches the test set.

Usage in Colab notebook:
    from src.hyperparam_search import run_search
    best = run_search(train_df, val_df, emotion_df)
    print(f"Best config: {best}")
"""

import pandas as pd
import numpy as np
from itertools import product
from src.transformer_model import MultiTaskClassifier, FAST_EPOCHS, FAST_BATCH


# ── Search grid ───────────────────────────────────────────────────────────────
# Top 3 learning rates from BERT paper
LEARNING_RATES = [5e-5, 3e-5, 2e-5]

# 3 loss weight combos: current default + two downweighted emotionality options
# Format: (ideology_weight, emotionality_weight)
LOSS_WEIGHTS = [
    (1.0, 1.0),   # current default — equal weighting
    (2.0, 0.5),   # downweight emotionality moderately
    (1.0, 0.25),  # very low emotionality influence
]


def _train_and_eval(train_df, val_df, emotion_df, lr, ideo_weight, sens_weight,
                    epochs=3, batch_size=FAST_BATCH):
    """
    Train one model configuration and return val metrics.
    Modifies loss weighting by monkey-patching the train loop.
    """
    import torch
    import torch.nn as nn
    from itertools import cycle
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    from tqdm import tqdm
    from src.transformer_model import MultiTaskDistilBERT, MultiTaskDataset, DEVICE, MODEL_NAME
    from torch.utils.data import DataLoader

    # Build model fresh each run
    clf = MultiTaskClassifier()

    use_emotion = emotion_df is not None and len(emotion_df) >= 2

    ideology_loader = clf._make_ideology_loader(train_df, batch_size, shuffle=True)
    emotion_iter = cycle(clf._make_emotion_loader(emotion_df, batch_size)) if use_emotion else None

    ce_loss  = nn.CrossEntropyLoss()
    bce_loss = nn.BCELoss()

    optimizer = AdamW(clf.model.parameters(), lr=lr, weight_decay=0.01)
    total_steps  = len(ideology_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_val_f1 = 0.0
    best_epoch  = 0

    clf.model.train()
    for epoch in range(1, epochs + 1):
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(ideology_loader,
                    desc=f"    LR={lr:.0e} W={ideo_weight}:{sens_weight} Epoch {epoch}/{epochs}",
                    leave=False)

        for batch in pbar:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            ideo_labels    = batch["ideology_label"].to(DEVICE)

            ideo_logits, _ = clf.model(input_ids, attention_mask)
            loss = ideo_weight * ce_loss(ideo_logits, ideo_labels)

            if use_emotion:
                em_batch  = next(emotion_iter)
                em_ids    = em_batch["input_ids"].to(DEVICE)
                em_mask   = em_batch["attention_mask"].to(DEVICE)
                em_labels = em_batch["emotion_label"].to(DEVICE)
                _, em_pred = clf.model(em_ids, em_mask)
                loss = loss + sens_weight * bce_loss(em_pred, em_labels)

            optimizer.zero_grad()
            loss.backward()
            import torch as _torch
            _torch.nn.utils.clip_grad_norm_(clf.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * ideo_labels.size(0)
            preds    = ideo_logits.argmax(dim=-1)
            correct += (preds == ideo_labels).sum().item()
            total   += ideo_labels.size(0)

        # Evaluate on val set each epoch
        val_metrics = clf.evaluate_ideology(val_df)
        val_f1 = val_metrics["macro_f1"]
        print(f"    Epoch {epoch} — loss: {running_loss/total:.4f} | "
              f"val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch

    return {
        "lr":           lr,
        "ideo_weight":  ideo_weight,
        "sens_weight":  sens_weight,
        "best_val_f1":  best_val_f1,
        "best_epoch":   best_epoch,
    }


def run_search(train_df, val_df, emotion_df,
               epochs=3, batch_size=FAST_BATCH,
               learning_rates=None, loss_weights=None):
    """
    Run full hyperparameter search over learning rates x loss weights.
    All decisions made on val set — test set never touched.

    Args:
        train_df:      ideology training DataFrame
        val_df:        ideology validation DataFrame
        emotion_df:    emotionality labels DataFrame
        epochs:        epochs per config (3 is fine for search)
        batch_size:    batch size
        learning_rates: list of LRs to try (defaults to LEARNING_RATES)
        loss_weights:   list of (ideo_w, sens_w) tuples (defaults to LOSS_WEIGHTS)

    Returns:
        dict with best config and full results table
    """
    lrs     = learning_rates or LEARNING_RATES
    weights = loss_weights   or LOSS_WEIGHTS

    configs = list(product(lrs, weights))
    total   = len(configs)

    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER SEARCH")
    print(f"  {len(lrs)} learning rates × {len(weights)} loss weight configs")
    print(f"  = {total} total runs × {epochs} epochs each")
    print(f"  Emotionality labels: {len(emotion_df) if emotion_df is not None else 0}")
    print(f"{'='*60}\n")

    all_results = []

    for run_idx, (lr, (ideo_w, sens_w)) in enumerate(configs, 1):
        print(f"\n[{run_idx}/{total}] LR={lr:.0e} | "
              f"Loss weights: ideology={ideo_w}, emotionality={sens_w}")

        result = _train_and_eval(
            train_df, val_df, emotion_df,
            lr=lr, ideo_weight=ideo_w, sens_weight=sens_w,
            epochs=epochs, batch_size=batch_size,
        )
        all_results.append(result)

        print(f"  → Best val F1: {result['best_val_f1']:.4f} "
              f"(epoch {result['best_epoch']})")

    # Sort by val F1
    all_results.sort(key=lambda x: x["best_val_f1"], reverse=True)
    best = all_results[0]

    # Print summary table
    print(f"\n{'='*60}")
    print(f"SEARCH COMPLETE — RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'LR':<10} {'Ideo W':<8} {'Sens W':<8} {'Val F1':<10} {'Best Epoch'}")
    print(f"{'-'*50}")
    for r in all_results:
        marker = " ← BEST" if r == best else ""
        print(f"{r['lr']:<10.0e} {r['ideo_weight']:<8} {r['sens_weight']:<8} "
              f"{r['best_val_f1']:<10.4f} {r['best_epoch']}{marker}")

    print(f"\n{'='*60}")
    print(f"RECOMMENDATION:")
    print(f"  Learning rate:      {best['lr']:.0e}")
    print(f"  Ideology weight:    {best['ideo_weight']}")
    print(f"  Emotionality weight:{best['sens_weight']}")
    print(f"  Val F1:             {best['best_val_f1']:.4f}")
    print(f"\nNow run the final pipeline with these settings.")
    print(f"{'='*60}\n")

    return {
        "best":        best,
        "all_results": all_results,
        "results_df":  pd.DataFrame(all_results).sort_values(
                           "best_val_f1", ascending=False
                       ).reset_index(drop=True),
    }
