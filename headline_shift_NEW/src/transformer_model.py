"""
Transformer-based ideology classifier using DistilBERT.
Fine-tunes on the QBias dataset for left/center/right classification.

Also provides:
  - MultiTaskClassifier: dual-head DistilBERT (ideology + emotionality) — primary model
  - RoBERTaClassifier:   single-task RoBERTa (ideology only)         — Baseline 2
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from itertools import cycle
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizer,
    DistilBertModel,
    DistilBertForSequenceClassification,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    cohen_kappa_score, roc_auc_score,
)
from tqdm import tqdm
from src.utils import MODELS_DIR, IDEOLOGY_LABELS, IDEOLOGY_ID2LABEL, SEED, set_seed

set_seed()

MODEL_DIR = os.path.join(MODELS_DIR, "distilbert_ideology")
MULTITASK_DIR = os.path.join(MODELS_DIR, "distilbert_multitask")
ROBERTA_DIR = os.path.join(MODELS_DIR, "roberta_ideology")
MODEL_NAME = "distilbert-base-uncased"
ROBERTA_MODEL_NAME = "roberta-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Speed settings ────────────────────────────────────────────────────────
# Headlines are 7-20 tokens. max_len=128 wastes 6-8x compute per batch.
# 32 covers 99%+ of headlines with no accuracy loss.
FAST_MAX_LEN = 32    # use this everywhere instead of 128
FAST_BATCH   = 64    # larger batches = fewer gradient steps = faster
FAST_EPOCHS  = 1     # for prototype/mid-review; set to 3-5 for final run


class HeadlineDataset(Dataset):
    """Simple PyTorch dataset wrapping tokenized headlines + labels."""

    def __init__(self, texts, labels, tokenizer, max_len=FAST_MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class TransformerClassifier:
    """DistilBERT fine-tuned for ideology classification."""

    def __init__(self, model_name=MODEL_NAME, num_labels=3, max_len=FAST_MAX_LEN):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.model.to(DEVICE)
        self.max_len = max_len

    def _make_loader(self, texts, labels, batch_size, shuffle=False):
        ds = HeadlineDataset(texts, labels, self.tokenizer, self.max_len)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame = None,
        epochs: int = FAST_EPOCHS,
        batch_size: int = FAST_BATCH,
        lr: float = 2e-5,
    ):
        """Fine-tune the model. Prints val metrics each epoch if val_df given."""
        train_loader = self._make_loader(
            train_df["headline"].tolist(), train_df["label_id"].tolist(),
            batch_size, shuffle=True,
        )

        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        self.model.train()
        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            correct = 0
            total = 0

            pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{epochs}")
            for batch in pbar:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                running_loss += loss.item() * labels.size(0)
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

            epoch_loss = running_loss / total
            epoch_acc = correct / total
            print(f"  Epoch {epoch} — loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}")

            if val_df is not None:
                val_metrics = self.evaluate(val_df, batch_size)
                print(f"  Val acc: {val_metrics['accuracy']:.4f}  |  Val F1: {val_metrics['macro_f1']:.4f}")

    @torch.no_grad()
    def predict(self, texts, batch_size=64) -> np.ndarray:
        """Return predicted label IDs for a list of texts."""
        self.model.eval()
        dummy_labels = [0] * len(texts)
        loader = self._make_loader(texts, dummy_labels, batch_size)
        all_preds = []
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            logits = self.model(input_ids, attention_mask=attention_mask).logits
            all_preds.append(logits.argmax(dim=-1).cpu().numpy())
        return np.concatenate(all_preds)

    @torch.no_grad()
    def predict_proba(self, texts, batch_size=64) -> np.ndarray:
        """Return softmax probabilities for each class."""
        self.model.eval()
        dummy_labels = [0] * len(texts)
        loader = self._make_loader(texts, dummy_labels, batch_size)
        all_probs = []
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            logits = self.model(input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
        return np.concatenate(all_probs)

    def evaluate(self, df: pd.DataFrame, batch_size=64) -> dict:
        """Evaluate on a labeled dataframe, return metrics."""
        preds = self.predict(df["headline"].tolist(), batch_size)
        y_true = df["label_id"].values
        acc = (preds == y_true).mean()
        macro_f1 = f1_score(y_true, preds, average="macro")
        kappa = cohen_kappa_score(y_true, preds)
        cm = confusion_matrix(y_true, preds)
        report_str = classification_report(y_true, preds, target_names=IDEOLOGY_LABELS)
        report_dict = classification_report(y_true, preds, target_names=IDEOLOGY_LABELS, output_dict=True)
        return {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "cohen_kappa": kappa,
            "confusion_matrix": cm,
            "classification_report_str": report_str,
            "report": report_dict,
        }

    def save(self, path=MODEL_DIR):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"  Saved transformer model → {path}")

    @classmethod
    def load(cls, path=MODEL_DIR):
        obj = cls.__new__(cls)
        obj.tokenizer = DistilBertTokenizer.from_pretrained(path)
        obj.model = DistilBertForSequenceClassification.from_pretrained(path)
        obj.model.to(DEVICE)
        obj.model.eval()
        obj.max_len = 128
        return obj


def train_transformer(train_df, val_df, test_df, epochs=3, batch_size=32):
    """Convenience: train, evaluate, save. Returns model + test metrics."""
    print("\n══ Transformer: DistilBERT Fine-tuning ══")
    print(f"  Device: {DEVICE}")
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    clf = TransformerClassifier()
    clf.train(train_df, val_df, epochs=epochs, batch_size=batch_size)

    print("\n  ── Test set results ──")
    test_metrics = clf.evaluate(test_df)
    print(test_metrics["classification_report_str"])
    print("  Confusion matrix:")
    print(test_metrics["confusion_matrix"])

    clf.save()
    return clf, test_metrics


# ── Multi-task DistilBERT ──────────────────────────────────────────────────

class MultiTaskDistilBERT(nn.Module):
    """
    DistilBERT backbone with two task-specific heads:
      - Ideology head  : 768 → 256 → 3  (left / center / right)
      - Emotionality head: 768 → 128 → 1 (sigmoid, 0–1 score)
    """

    def __init__(self, model_name=MODEL_NAME, dropout=0.1):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(model_name)

        # Ideology head (deeper, 3-class)
        self.ideology_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 3),
        )

        # Emotionality head (shallower, scalar sigmoid)
        self.emotion_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids, attention_mask):
        cls = self.distilbert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.ideology_head(cls), self.emotion_head(cls).squeeze(-1)


class MultiTaskDataset(Dataset):
    """Headlines with ideology label and optional emotionality label."""

    def __init__(self, texts, ideology_labels, tokenizer, max_len=FAST_MAX_LEN, emotion_labels=None):
        self.texts = texts
        self.ideology_labels = ideology_labels
        self.emotion_labels = emotion_labels  # None or list of floats
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "ideology_label": torch.tensor(self.ideology_labels[idx], dtype=torch.long),
        }
        if self.emotion_labels is not None:
            item["emotion_label"] = torch.tensor(float(self.emotion_labels[idx]), dtype=torch.float)
        return item


class MultiTaskClassifier:
    """
    Dual-head DistilBERT: jointly predicts ideology (3-class) + emotionality (scalar).

    If no emotionality labels are provided, trains ideology-only (ablation mode).
    This is the primary model described in the project proposal.
    """

    def __init__(self, model_name=MODEL_NAME, max_len=FAST_MAX_LEN):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = MultiTaskDistilBERT(model_name)
        self.model.to(DEVICE)
        self.max_len = max_len
        self._has_emotion_head = True  # always built; only used when labels provided

    def _make_ideology_loader(self, df, batch_size, shuffle=False):
        ds = MultiTaskDataset(
            df["headline"].tolist(), df["label_id"].tolist(),
            self.tokenizer, self.max_len,
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    def _make_emotion_loader(self, emotion_df, batch_size):
        """emotion_df must have columns: headline, emotionality_score."""
        # Use dummy ideology labels (0) — only the emotion loss is computed from this loader
        dummy_labels = [0] * len(emotion_df)
        ds = MultiTaskDataset(
            emotion_df["headline"].tolist(), dummy_labels,
            self.tokenizer, self.max_len,
            emotion_labels=emotion_df["emotionality_score"].tolist(),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=True)

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame = None,
        emotion_df: pd.DataFrame = None,
        epochs: int = FAST_EPOCHS,
        batch_size: int = FAST_BATCH,
        lr: float = 2e-5,
    ):
        """
        Fine-tune the model.
        - emotion_df: optional DataFrame with [headline, emotionality_score] columns.
          If None, trains ideology-only (ablation mode).
        """
        use_emotion = emotion_df is not None and len(emotion_df) >= 2
        if not use_emotion:
            print("  ⚠  No emotionality labels provided — training ideology-only (ablation mode).")

        ideology_loader = self._make_ideology_loader(train_df, batch_size, shuffle=True)
        emotion_iter = cycle(self._make_emotion_loader(emotion_df, batch_size)) if use_emotion else None

        ce_loss = nn.CrossEntropyLoss()
        bce_loss = nn.BCELoss()

        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(ideology_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        self.model.train()
        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            correct = 0
            total = 0

            pbar = tqdm(ideology_loader, desc=f"  Epoch {epoch}/{epochs}")
            for batch in pbar:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                ideology_labels = batch["ideology_label"].to(DEVICE)

                ideology_logits, emotion_pred = self.model(input_ids, attention_mask)
                loss = ce_loss(ideology_logits, ideology_labels)

                if use_emotion:
                    em_batch = next(emotion_iter)
                    em_ids = em_batch["input_ids"].to(DEVICE)
                    em_mask = em_batch["attention_mask"].to(DEVICE)
                    em_labels = em_batch["emotion_label"].to(DEVICE)
                    _, em_pred = self.model(em_ids, em_mask)
                    loss = loss + bce_loss(em_pred, em_labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                running_loss += loss.item() * ideology_labels.size(0)
                preds = ideology_logits.argmax(dim=-1)
                correct += (preds == ideology_labels).sum().item()
                total += ideology_labels.size(0)
                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

            print(f"  Epoch {epoch} — loss: {running_loss/total:.4f}, acc: {correct/total:.4f}")
            if val_df is not None:
                val_m = self.evaluate_ideology(val_df, batch_size)
                print(f"  Val acc: {val_m['accuracy']:.4f}  |  Val F1: {val_m['macro_f1']:.4f}")

    @torch.no_grad()
    def predict_ideology(self, texts, batch_size=64) -> np.ndarray:
        """Return predicted ideology label IDs."""
        self.model.eval()
        dummy = [0] * len(texts)
        ds = MultiTaskDataset(texts, dummy, self.tokenizer, self.max_len)
        loader = DataLoader(ds, batch_size=batch_size)
        preds = []
        for batch in loader:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            logits, _ = self.model(ids, mask)
            preds.append(logits.argmax(dim=-1).cpu().numpy())
        return np.concatenate(preds)

    @torch.no_grad()
    def predict_proba_ideology(self, texts, batch_size=64) -> np.ndarray:
        """Return softmax probabilities for ideology classes."""
        self.model.eval()
        dummy = [0] * len(texts)
        ds = MultiTaskDataset(texts, dummy, self.tokenizer, self.max_len)
        loader = DataLoader(ds, batch_size=batch_size)
        probs = []
        for batch in loader:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            logits, _ = self.model(ids, mask)
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(probs)

    @torch.no_grad()
    def predict_emotionality(self, texts, batch_size=64) -> np.ndarray:
        """Return emotionality scores in [0, 1]."""
        self.model.eval()
        dummy = [0] * len(texts)
        ds = MultiTaskDataset(texts, dummy, self.tokenizer, self.max_len)
        loader = DataLoader(ds, batch_size=batch_size)
        scores = []
        for batch in loader:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            _, em = self.model(ids, mask)
            scores.append(em.cpu().numpy())
        return np.concatenate(scores)

    def evaluate_ideology(self, df: pd.DataFrame, batch_size=64) -> dict:
        """Ideology metrics: accuracy, macro F1, Cohen's Kappa."""
        preds = self.predict_ideology(df["headline"].tolist(), batch_size)
        y_true = df["label_id"].values
        acc = (preds == y_true).mean()
        macro_f1 = f1_score(y_true, preds, average="macro")
        kappa = cohen_kappa_score(y_true, preds)
        cm = confusion_matrix(y_true, preds)
        report_str = classification_report(y_true, preds, target_names=IDEOLOGY_LABELS)
        return {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "cohen_kappa": kappa,
            "confusion_matrix": cm,
            "classification_report_str": report_str,
        }

    def evaluate_emotionality(self, emotion_df: pd.DataFrame, batch_size=64) -> dict:
        """
        Emotionality metrics: F1 (threshold=0.5) + AUC-ROC.
        emotion_df must have [headline, emotionality_score] columns.
        Binary labels are created by median split of the true scores.
        """
        scores = self.predict_emotionality(emotion_df["headline"].tolist(), batch_size)
        y_true_cont = emotion_df["emotionality_score"].values
        threshold = np.median(y_true_cont)
        y_true_bin = (y_true_cont >= threshold).astype(int)
        y_pred_bin = (scores >= 0.5).astype(int)
        f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        try:
            auc = roc_auc_score(y_true_bin, scores)
        except ValueError:
            auc = float("nan")
        return {"f1": f1, "auc_roc": auc}

    def save(self, path=MULTITASK_DIR):
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))
        self.tokenizer.save_pretrained(path)
        print(f"  Saved multi-task model → {path}")

    @classmethod
    def load(cls, path=MULTITASK_DIR, model_name=MODEL_NAME):
        obj = cls.__new__(cls)
        obj.tokenizer = DistilBertTokenizer.from_pretrained(path)
        obj.model = MultiTaskDistilBERT(model_name)
        obj.model.load_state_dict(torch.load(os.path.join(path, "model.pt"), map_location=DEVICE))
        obj.model.to(DEVICE)
        obj.model.eval()
        obj.max_len = 128
        return obj


def train_multitask(train_df, val_df, test_df, emotion_df=None, epochs=3, batch_size=32):
    """Convenience: train MultiTaskClassifier, print results, save. Returns clf + metrics."""
    label = "Multi-task DistilBERT" if emotion_df is not None and len(emotion_df) >= 2 else "DistilBERT Ablation (ideology-only)"
    print(f"\n══ {label} ══")
    print(f"  Device: {DEVICE}")
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    if emotion_df is not None:
        print(f"  Emotionality labels: {len(emotion_df)}")

    clf = MultiTaskClassifier()
    clf.train(train_df, val_df, emotion_df=emotion_df, epochs=epochs, batch_size=batch_size)

    print("\n  ── Ideology test results ──")
    test_metrics = clf.evaluate_ideology(test_df)
    print(test_metrics["classification_report_str"])
    print("  Confusion matrix:")
    print(test_metrics["confusion_matrix"])
    print(f"  Cohen's Kappa: {test_metrics['cohen_kappa']:.4f}")

    if emotion_df is not None and len(emotion_df) >= 2:
        print("\n  ── Emotionality test results ──")
        em_metrics = clf.evaluate_emotionality(emotion_df)
        print(f"  F1: {em_metrics['f1']:.4f}  |  AUC-ROC: {em_metrics['auc_roc']:.4f}")
        test_metrics["emotionality"] = em_metrics

    clf.save()
    return clf, test_metrics


# ── RoBERTa single-task (Baseline 2) ──────────────────────────────────────

class RoBERTaClassifier:
    """
    Fine-tuned RoBERTa-base for ideology classification (single-task, Baseline 2).
    Same interface as TransformerClassifier.
    """

    def __init__(self, model_name=ROBERTA_MODEL_NAME, num_labels=3, max_len=FAST_MAX_LEN):
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.model.to(DEVICE)
        self.max_len = max_len

    def _make_loader(self, texts, labels, batch_size, shuffle=False):
        ds = HeadlineDataset(texts, labels, self.tokenizer, self.max_len)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    def train(self, train_df, val_df=None, epochs=FAST_EPOCHS, batch_size=FAST_BATCH, lr=2e-5):
        train_loader = self._make_loader(
            train_df["headline"].tolist(), train_df["label_id"].tolist(),
            batch_size, shuffle=True,
        )
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )
        self.model.train()
        for epoch in range(1, epochs + 1):
            running_loss, correct, total = 0.0, 0, 0
            pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{epochs}")
            for batch in pbar:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                running_loss += loss.item() * labels.size(0)
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")
            print(f"  Epoch {epoch} — loss: {running_loss/total:.4f}, acc: {correct/total:.4f}")
            if val_df is not None:
                val_m = self.evaluate(val_df, batch_size)
                print(f"  Val acc: {val_m['accuracy']:.4f}  |  Val F1: {val_m['macro_f1']:.4f}")

    @torch.no_grad()
    def predict(self, texts, batch_size=64) -> np.ndarray:
        self.model.eval()
        dummy = [0] * len(texts)
        loader = self._make_loader(texts, dummy, batch_size)
        preds = []
        for batch in loader:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            logits = self.model(ids, attention_mask=mask).logits
            preds.append(logits.argmax(dim=-1).cpu().numpy())
        return np.concatenate(preds)

    @torch.no_grad()
    def predict_proba(self, texts, batch_size=64) -> np.ndarray:
        self.model.eval()
        dummy = [0] * len(texts)
        loader = self._make_loader(texts, dummy, batch_size)
        probs = []
        for batch in loader:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            logits = self.model(ids, attention_mask=mask).logits
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(probs)

    def evaluate(self, df: pd.DataFrame, batch_size=64) -> dict:
        preds = self.predict(df["headline"].tolist(), batch_size)
        y_true = df["label_id"].values
        acc = (preds == y_true).mean()
        macro_f1 = f1_score(y_true, preds, average="macro")
        kappa = cohen_kappa_score(y_true, preds)
        cm = confusion_matrix(y_true, preds)
        report_str = classification_report(y_true, preds, target_names=IDEOLOGY_LABELS)
        return {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "cohen_kappa": kappa,
            "confusion_matrix": cm,
            "classification_report_str": report_str,
        }

    def save(self, path=ROBERTA_DIR):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"  Saved RoBERTa model → {path}")

    @classmethod
    def load(cls, path=ROBERTA_DIR):
        obj = cls.__new__(cls)
        obj.tokenizer = RobertaTokenizer.from_pretrained(path)
        obj.model = RobertaForSequenceClassification.from_pretrained(path)
        obj.model.to(DEVICE)
        obj.model.eval()
        obj.max_len = 128
        return obj


def train_roberta(train_df, val_df, test_df, epochs=3, batch_size=32):
    """Convenience: train RoBERTa baseline, print results, save. Returns clf + metrics."""
    print("\n══ Baseline 2: RoBERTa-base (single-task ideology) ══")
    print(f"  Device: {DEVICE}")
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    clf = RoBERTaClassifier()
    clf.train(train_df, val_df, epochs=epochs, batch_size=batch_size)

    print("\n  ── Test set results ──")
    test_metrics = clf.evaluate(test_df)
    print(test_metrics["classification_report_str"])
    print("  Confusion matrix:")
    print(test_metrics["confusion_matrix"])
    print(f"  Cohen's Kappa: {test_metrics['cohen_kappa']:.4f}")

    clf.save()
    return clf, test_metrics
