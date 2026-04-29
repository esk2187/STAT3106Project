"""
Baseline ideology classifier: TF-IDF + Logistic Regression.
Simple, fast, and surprisingly decent for text classification.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from src.utils import MODELS_DIR, IDEOLOGY_LABELS, set_seed

set_seed()

MODEL_PATH = os.path.join(MODELS_DIR, "baseline_logreg.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "baseline_tfidf.joblib")


class BaselineClassifier:
    """TF-IDF + Logistic Regression for left/center/right classification."""

    def __init__(self, max_features: int = 30_000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            sublinear_tf=True,  # dampens term frequency – helps with headline-length text
        )
        self.model = LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs",
            multi_class="multinomial", random_state=42,
        )
        self._fitted = False

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None):
        """Fit vectorizer + LR on training data. Optionally print val metrics."""
        X_train = self.vectorizer.fit_transform(train_df["headline"])
        y_train = train_df["label_id"]
        self.model.fit(X_train, y_train)
        self._fitted = True

        # training accuracy
        train_acc = self.model.score(X_train, y_train)
        print(f"  Train accuracy: {train_acc:.4f}")

        if val_df is not None:
            val_metrics = self.evaluate(val_df)
            print(f"  Val accuracy:   {val_metrics['accuracy']:.4f}")
            print(f"  Val macro-F1:   {val_metrics['macro_f1']:.4f}")

    def predict(self, texts) -> np.ndarray:
        """Return predicted label IDs."""
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)

    def predict_proba(self, texts) -> np.ndarray:
        """Return probability distributions over classes."""
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)

    def evaluate(self, df: pd.DataFrame) -> dict:
        """Run predictions, return metrics dict + print classification report."""
        preds = self.predict(df["headline"])
        y_true = df["label_id"].values

        report = classification_report(
            y_true, preds, target_names=IDEOLOGY_LABELS, output_dict=True
        )
        cm = confusion_matrix(y_true, preds)
        acc = (preds == y_true).mean()
        macro_f1 = f1_score(y_true, preds, average="macro")

        return {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "report": report,
            "confusion_matrix": cm,
            "classification_report_str": classification_report(
                y_true, preds, target_names=IDEOLOGY_LABELS
            ),
        }

    def save(self, model_path=MODEL_PATH, vec_path=VECTORIZER_PATH):
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vec_path)
        print(f"  Saved model → {model_path}")
        print(f"  Saved vectorizer → {vec_path}")

    @classmethod
    def load(cls, model_path=MODEL_PATH, vec_path=VECTORIZER_PATH):
        obj = cls.__new__(cls)
        obj.model = joblib.load(model_path)
        obj.vectorizer = joblib.load(vec_path)
        obj._fitted = True
        return obj


def train_baseline(train_df, val_df, test_df):
    """Convenience: train, evaluate, save, return metrics."""
    print("\n══ Baseline: TF-IDF + Logistic Regression ══")
    clf = BaselineClassifier()
    clf.train(train_df, val_df)

    print("\n  ── Test set results ──")
    test_metrics = clf.evaluate(test_df)
    print(test_metrics["classification_report_str"])
    print("  Confusion matrix:")
    print(test_metrics["confusion_matrix"])

    clf.save()
    return clf, test_metrics
