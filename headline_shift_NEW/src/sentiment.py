"""
Sentiment and emotionality scoring for headlines.
Uses VADER (rule-based, tuned for social media / short text) as primary scorer.
Optionally layers in a HuggingFace transformer sentiment model.
"""

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# try to import the transformer sentiment pipeline; optional dependency
try:
    from transformers import pipeline as hf_pipeline
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


class SentimentScorer:
    """
    Multi-method sentiment scorer for headlines.
    
    Methods:
        vader   – fast, no GPU needed, good for short text
        hf      – HuggingFace distilbert-sst2 (optional, more accurate on formal text)
    """

    def __init__(self, use_hf: bool = False):
        self.vader = SentimentIntensityAnalyzer()
        self.hf_pipe = None
        if use_hf and _HF_AVAILABLE:
            print("  Loading HuggingFace sentiment model …")
            self.hf_pipe = hf_pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1,  # CPU
                truncation=True,
                max_length=128,
            )

    def score_vader(self, text: str) -> dict:
        """Return VADER compound + pos/neg/neu scores."""
        scores = self.vader.polarity_scores(text)
        return {
            "sentiment_score": scores["compound"],        # -1 to +1
            "sentiment_pos": scores["pos"],
            "sentiment_neg": scores["neg"],
            "sentiment_neu": scores["neu"],
        }

    def score_hf(self, text: str) -> dict:
        """Score using HuggingFace sentiment pipeline."""
        if self.hf_pipe is None:
            return {"hf_sentiment": None, "hf_confidence": None}
        result = self.hf_pipe(text[:512])[0]
        # convert POSITIVE/NEGATIVE to numeric
        score = result["score"] if result["label"] == "POSITIVE" else -result["score"]
        return {
            "hf_sentiment": score,
            "hf_confidence": result["score"],
        }

    def score_emotionality(self, text: str) -> float:
        """
        Simple emotionality proxy: absolute value of VADER compound.
        Higher = more emotionally charged (regardless of direction).
        """
        compound = self.vader.polarity_scores(text)["compound"]
        return abs(compound)

    def score_batch(self, texts, use_hf: bool = False) -> pd.DataFrame:
        """Score a list of texts, return a DataFrame of scores."""
        records = []
        for text in texts:
            row = self.score_vader(str(text))
            row["emotionality"] = abs(row["sentiment_score"])
            if use_hf and self.hf_pipe is not None:
                row.update(self.score_hf(str(text)))
            records.append(row)
        return pd.DataFrame(records)

    def score_dataframe(self, df: pd.DataFrame, text_col: str = "headline") -> pd.DataFrame:
        """Add sentiment columns to an existing dataframe (in-place friendly)."""
        scores_df = self.score_batch(df[text_col].tolist())
        # assign confidence as absolute value
        scores_df["sentiment_confidence"] = scores_df["sentiment_score"].abs()
        return pd.concat([df.reset_index(drop=True), scores_df], axis=1)
