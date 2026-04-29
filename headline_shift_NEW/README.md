## Project Structure

```
headline_shift/
├── README.md                    ← you are here
├── requirements.txt             # all dependencies
├── run_pipeline.py              # end-to-end pipeline script
├── data/
│   ├── download_data.py         # dataset download & preparation
│   ├── qbias/                   # AllSides dataset
│   └── processed/               # cleaned CSVs
├── models/                      # saved model weights
├── notebooks/
│   └── analysis.ipynb           # interactive analysis notebook
├── src/
│   ├── __init__.py
│   ├── utils.py                 # shared constants & helpers
│   ├── data_loader.py           # data loading & splitting
│   ├── baseline_model.py        # TF-IDF + Logistic Regression
│   ├── transformer_model.py     # DistilBERT + RoBERTa fine-tuning (single + dual-head)
│   ├── hyperparam_search.py     # validation-set hyperparameter search
│   ├── sentiment.py             # VADER sentiment scoring
│   ├── inference.py             # full scoring pipeline
│   └── time_series.py           # trend analysis & visualization
├── app/
│   └── active_learning_app.py   # Streamlit crowdsourcing UI (backed by Supabase)
└── outputs/
    ├── plots/                   # generated visualizations (PNG)
    └── results/                 # inference output CSVs
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

```bash
python data/download_data.py
```

This will:
- Process the AllSides dataset (21,754 labeled headlines with left/center/right ideology)
- Download the Kaggle 4.5M headline dataset via kagglehub
- Filter to CNN, Fox News, Washington Post, New York Times (2013–2022)

### 3. Run the full pipeline

```bash
# Full pipeline — dual-head DistilBERT (primary model)
python run_pipeline.py --epochs 5 --emotionality-labels data/processed/emotionality_labels.csv

# DistilBERT ideology-only (ablation)
python run_pipeline.py --epochs 5 --model ablation

# RoBERTa single-task (Baseline 2 — best performing model, F1=0.47)
python run_pipeline.py --epochs 5 --model roberta

# RoBERTa dual-head (extended comparison)
python run_pipeline.py --epochs 5 --model roberta_multitask --emotionality-labels data/processed/emotionality_labels.csv

# Train only, skip inference and plots
python run_pipeline.py --epochs 5 --model roberta --no-inference

# TF-IDF baseline only (fastest)
python run_pipeline.py --skip-transformer

# Just prepare data
python run_pipeline.py --data-only
```

### 4. Run hyperparameter search

```python
from src.data_loader import load_qbias, split_qbias
from src.hyperparam_search import run_search
import pandas as pd

qbias = load_qbias()
train_df, val_df, test_df = split_qbias(qbias)
em_df = pd.read_csv('data/processed/emotionality_labels.csv')

# Searches 9 configs (3 LRs x 3 loss weights) on validation set only
results = run_search(train_df, val_df, em_df, epochs=3)
```

### 5. Launch the annotation app

```bash
streamlit run app/active_learning_app.py
```

Or visit the live app at: https://esk2187-mlapp.hf.space

## Datasets

| Dataset | Source | Size | Purpose |
|---------|--------|------|---------|
| AllSides Balanced News | [GitHub](https://github.com/irgroup/Qbias) | 21,754 headlines | Ideology labels (left/center/right) for training |
| Headlines 2007–2022 | [Kaggle](https://www.kaggle.com/datasets/jordankrishnayah/45m-headlines-from-2007-2022-10-largest-sites) | 1,068,680 headlines (filtered) | Main inference corpus for time-series analysis |
| Emotionality Labels | Crowdsourced via annotation app | 1,095 pairwise comparisons | Emotionality supervision signal for dual-head models |

## Models

All models use a train/val/test split of 15,227 / 3,263 / 3,264 (stratified, random_state=42).
Hyperparameters (LR=3e-5, loss weights ideology:2.0 / emotionality:0.5) selected via 9-configuration validation set search.

| Model | Test Macro F1 | Kappa | Notes |
|-------|--------------|-------|-------|
| TF-IDF + Logistic Regression | 0.33 | — | Non-neural baseline |
| DistilBERT Ablation (ideology-only) | 0.43 | 0.14 | Single-task, no emotionality head |
| DistilBERT Dual-Head | 0.42 | 0.14 | Joint ideology + emotionality |
| RoBERTa Single-Task | **0.47** | **0.21** | Best model — Baseline 2 |
| RoBERTa Dual-Head | 0.45 | 0.18 | Extended comparison |

**Key finding:** Multi-task training consistently hurts ideology F1 by ~0.02 points regardless of backbone. Backbone quality (RoBERTa > DistilBERT) is the dominant factor. The multi-task hypothesis was not supported at our annotation scale.

### Architecture

**Dual-head models:** DistilBERT/RoBERTa backbone with [CLS] token → ideology head (768→256→3, softmax) + emotionality head (768→128→1, sigmoid). Joint loss: 2.0 × CrossEntropy(ideology) + 0.5 × BCE(emotionality).

### Sentiment: VADER
Rule-based sentiment analyzer tuned for short text. Compound score: -1 to +1. Emotionality proxy: |compound|.

## Annotation App

The Streamlit app crowdsources emotionality labels via pairwise comparison:
- Shows pairs of headlines and asks which is more emotionally charged
- Annotations stored persistently in Supabase (PostgreSQL)
- Win rates converted to continuous emotionality scores
- Live at: https://esk2187-mlapp.hf.space

## Visualizations

The pipeline generates 7 plots saved to `outputs/plots/`:

1. **Ideology Trends** — quarterly ideology scores with 95% CI per publication
2. **Sentiment Trends** — quarterly VADER sentiment with CI
3. **Emotionality Trends** — VADER-based emotionality over time
4. **Ideology by Topic** — heatmap of mean ideology score by topic × publication
5. **Sentiment Distribution** — violin plots per publication
6. **Ideology Distribution** — stacked bars of ideology proportions by year
7. **Election Effect** — emotionality in election vs. non-election years

## Research Questions

1. Do CNN, Fox News, WaPo, and NYT headlines show measurable ideology shifts from 2013–2022?
2. Does joint emotionality supervision improve ideology classification? (Answer: No — robust negative result)
3. Does backbone quality or multi-task learning dominate performance? (Answer: Backbone quality)
4. Do ideology and sentiment patterns differ by topic area and election proximity?

## License

Academic use only. Datasets subject to their own licenses.
