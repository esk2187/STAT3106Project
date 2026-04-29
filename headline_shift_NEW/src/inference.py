"""
Full inference pipeline: run ideology + sentiment scoring on headline data.
Outputs a structured CSV ready for time-series analysis.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import (
    RESULTS_DIR, IDEOLOGY_LABELS, IDEOLOGY_ID2LABEL, label_to_numeric
)
from src.baseline_model import BaselineClassifier
from src.sentiment import SentimentScorer


def assign_topic(headline: str) -> str:
    """
    Simple keyword-based topic assignment.
    Not production-grade, but gives us topic_area for analysis.
    """
    headline_lower = headline.lower()
    topic_keywords = {
        "politics": ["election", "democrat", "republican", "congress", "senate", "vote",
                      "trump", "biden", "obama", "political", "partisan", "gop", "dnc",
                      "legislation", "lawmaker", "governor", "president", "campaign"],
        "economy": ["economy", "economic", "inflation", "jobs", "unemployment", "gdp",
                     "market", "stock", "trade", "tariff", "tax", "wage", "recession",
                     "federal reserve", "interest rate", "business"],
        "healthcare": ["health", "covid", "pandemic", "vaccine", "hospital", "medical",
                        "doctor", "insurance", "medicare", "medicaid", "drug", "pharma"],
        "immigration": ["immigration", "immigrant", "border", "migrant", "asylum",
                         "deportation", "refugee", "visa", "daca", "undocumented"],
        "climate": ["climate", "environment", "carbon", "emission", "renewable", "energy",
                     "fossil", "green", "warming", "weather", "hurricane", "wildfire"],
        "education": ["school", "education", "student", "teacher", "university", "college",
                       "campus", "learning", "curriculum"],
        "technology": ["tech", "technology", "ai", "artificial intelligence", "social media",
                        "facebook", "google", "apple", "amazon", "algorithm", "data",
                        "privacy", "cyber", "internet"],
        "crime": ["crime", "murder", "shooting", "gun", "police", "arrest", "prison",
                   "violence", "criminal", "homicide", "robbery", "assault"],
        "foreign_policy": ["china", "russia", "iran", "war", "military", "nato", "un",
                            "foreign", "diplomat", "sanction", "nuclear", "afghanistan",
                            "ukraine", "syria", "north korea"],
        "social_issues": ["race", "racial", "protest", "lgbtq", "gender", "equality",
                           "discrimination", "rights", "justice", "diversity", "abortion"],
    }
    for topic, keywords in topic_keywords.items():
        if any(kw in headline_lower for kw in keywords):
            return topic
    return "other"


def run_inference(
    headlines_df: pd.DataFrame,
    baseline_model=None,
    transformer_model=None,
    multitask_model=None,
    roberta_model=None,
    roberta_multitask_model=None,
    use_hf_sentiment: bool = False,
    output_path: str = None,
) -> pd.DataFrame:
    """
    Run the full scoring pipeline on headlines.

    Args:
        headlines_df:          DataFrame with columns [publication, date, headline]
        baseline_model:        trained BaselineClassifier (or None to load from disk)
        transformer_model:     trained TransformerClassifier / RoBERTaClassifier (or None)
        multitask_model:       trained MultiTaskClassifier — dual-head DistilBERT (or None)
        roberta_model:         trained RoBERTaClassifier — single-task (or None)
        roberta_multitask_model: trained MultiTaskRoBERTaClassifier — dual-head (or None)
        use_hf_sentiment:      whether to also run HF sentiment model
        output_path:           where to save the output CSV

    Returns:
        DataFrame with all scores appended
    """
    df = headlines_df.copy()

    print(f"\n══ Inference Pipeline ══")
    print(f"  Processing {len(df)} headlines …")

    # ── 1. Ideology scoring (baseline) ────────────────────────────────────
    if baseline_model is None:
        try:
            baseline_model = BaselineClassifier.load()
            print("  Loaded baseline model from disk.")
        except Exception as e:
            print(f"  ⚠ Could not load baseline model: {e}")

    if baseline_model is not None:
        print("  Running baseline ideology classifier …")
        preds = baseline_model.predict(df["headline"])
        probs = baseline_model.predict_proba(df["headline"])
        df["ideology_baseline"] = [IDEOLOGY_ID2LABEL[p] for p in preds]
        df["ideology_baseline_confidence"] = probs.max(axis=1)
        df["ideology_score"] = df["ideology_baseline"]
        df["ideology_confidence"] = df["ideology_baseline_confidence"]
    else:
        df["ideology_score"] = "unknown"
        df["ideology_confidence"] = 0.0

    # ── 2. Ideology scoring (single-task transformer, if available) ───────
    if transformer_model is not None:
        print("  Running transformer ideology classifier …")
        preds = transformer_model.predict(df["headline"].tolist())
        probs = transformer_model.predict_proba(df["headline"].tolist())
        df["ideology_transformer"] = [IDEOLOGY_ID2LABEL[p] for p in preds]
        df["ideology_transformer_confidence"] = probs.max(axis=1)
        df["ideology_score"] = df["ideology_transformer"]
        df["ideology_confidence"] = df["ideology_transformer_confidence"]

    # ── 3. Ideology + emotionality scoring (multi-task DistilBERT, if available)
    if multitask_model is not None:
        print("  Running multi-task ideology + emotionality classifier …")
        preds = multitask_model.predict_ideology(df["headline"].tolist())
        probs = multitask_model.predict_proba_ideology(df["headline"].tolist())
        df["ideology_multitask"] = [IDEOLOGY_ID2LABEL[p] for p in preds]
        df["ideology_multitask_confidence"] = probs.max(axis=1)
        df["ideology_score"] = df["ideology_multitask"]
        df["ideology_confidence"] = df["ideology_multitask_confidence"]

        em_scores = multitask_model.predict_emotionality(df["headline"].tolist())
        df["emotionality_model"] = em_scores

    # ── 4. RoBERTa single-task (if available) ─────────────────────────────
    if roberta_model is not None:
        print("  Running RoBERTa single-task ideology classifier …")
        preds = roberta_model.predict(df["headline"].tolist())
        probs = roberta_model.predict_proba(df["headline"].tolist())
        df["ideology_roberta"] = [IDEOLOGY_ID2LABEL[p] for p in preds]
        df["ideology_roberta_confidence"] = probs.max(axis=1)
        # RoBERTa single-task takes priority as best single-task model
        df["ideology_score"] = df["ideology_roberta"]
        df["ideology_confidence"] = df["ideology_roberta_confidence"]

    # ── 5. RoBERTa dual-head (if available) ───────────────────────────────
    if roberta_multitask_model is not None:
        print("  Running RoBERTa dual-head ideology + emotionality classifier …")
        preds = roberta_multitask_model.predict_ideology(df["headline"].tolist())
        probs_arr = []
        dummy = [0] * len(df)
        from src.transformer_model import MultiTaskDataset, DEVICE, FAST_MAX_LEN
        import torch
        from torch.utils.data import DataLoader
        tokenizer = roberta_multitask_model.tokenizer
        ds = MultiTaskDataset(df["headline"].tolist(), dummy, tokenizer, FAST_MAX_LEN)
        loader = DataLoader(ds, batch_size=64)
        all_probs = []
        roberta_multitask_model.model.eval()
        with torch.no_grad():
            for batch in loader:
                ids = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                logits, _ = roberta_multitask_model.model(ids, mask)
                all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        import numpy as _np
        probs = _np.concatenate(all_probs)
        df["ideology_roberta_multitask"] = [IDEOLOGY_ID2LABEL[p] for p in preds]
        df["ideology_roberta_multitask_confidence"] = probs.max(axis=1)
        df["ideology_score"] = df["ideology_roberta_multitask"]
        df["ideology_confidence"] = df["ideology_roberta_multitask_confidence"]

        em_scores = roberta_multitask_model.predict_emotionality(df["headline"].tolist())
        df["emotionality_roberta"] = em_scores

    # ── 6. Sentiment scoring (VADER) ──────────────────────────────────────
    print("  Running sentiment analysis …")
    scorer = SentimentScorer(use_hf=use_hf_sentiment)
    df = scorer.score_dataframe(df, text_col="headline")

    # ── 7. Topic assignment ───────────────────────────────────────────────
    print("  Assigning topic areas …")
    df["topic_area"] = df["headline"].apply(assign_topic)

    # ── 8. Numeric ideology for trends ────────────────────────────────────
    df["ideology_numeric"] = df["ideology_score"].apply(label_to_numeric)

    # ── 7. Save ───────────────────────────────────────────────────────────
    if output_path is None:
        output_path = os.path.join(RESULTS_DIR, "scored_headlines.csv")

    out_cols = [
        "publication", "date", "headline", "ideology_score", "ideology_confidence",
        "sentiment_score", "sentiment_confidence", "emotionality", "topic_area",
    ]
    for extra in [
        "ideology_baseline", "ideology_transformer", "ideology_multitask",
        "ideology_roberta", "ideology_roberta_multitask",
        "ideology_numeric",
        "ideology_baseline_confidence", "ideology_transformer_confidence",
        "ideology_multitask_confidence", "ideology_roberta_confidence",
        "ideology_roberta_multitask_confidence",
        "emotionality_model", "emotionality_roberta",
        "sentiment_pos", "sentiment_neg", "sentiment_neu",
        "year", "month", "quarter",
    ]:
        if extra in df.columns:
            out_cols.append(extra)

    out_cols = [c for c in out_cols if c in df.columns]
    df_out = df[out_cols]
    df_out.to_csv(output_path, index=False)
    print(f"  ✓ Saved {len(df_out)} scored headlines → {output_path}")

    return df
