# The Headline Shift: Measuring Ideology and Sentiment in U.S. Online News Headlines Over Time

Analyzing whether headlines from CNN, Fox News, Washington Post, and the New York Times show measurable shifts in political ideology and emotional phrasing from 2013–2022.

## Project Structure

```
headline_shift/
├── README.md                    ← you are here
├── requirements.txt             # all dependencies
├── run_pipeline.py              # end-to-end pipeline script
├── data/
│   ├── download_data.py         # dataset download & preparation
│   ├── qbias/                   # QBias repo (cloned)
│   └── processed/               # cleaned CSVs
├── models/                      # saved model weights
├── notebooks/
│   └── analysis.ipynb           # interactive analysis notebook
├── src/
│   ├── __init__.py
│   ├── utils.py                 # shared constants & helpers
│   ├── data_loader.py           # data loading & splitting
│   ├── baseline_model.py        # TF-IDF + Logistic Regression
│   ├── transformer_model.py     # DistilBERT fine-tuning
│   ├── sentiment.py             # VADER + optional HF sentiment
│   ├── inference.py             # full scoring pipeline
│   └── time_series.py           # trend analysis & visualization
├── app/
│   └── active_learning_app.py   # Streamlit crowdsourcing UI
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
- Process the QBias dataset (21K+ labeled headlines with left/center/right ideology)
- Attempt to download the Kaggle 4.5M headline dataset
- Fall back to synthetic sample data if Kaggle isn't available

**To use the real Kaggle dataset**, set up the Kaggle API:
```bash
pip install kaggle
# Place your kaggle.json in ~/.kaggle/
kaggle datasets download jordankrishnayah/45m-headlines-from-2007-2022-10-largest-sites -p data/
```

### 3. Run the full pipeline

```bash
# Full pipeline (baseline + transformer + inference + plots)
python run_pipeline.py

# Skip transformer training (faster, uses baseline only)
python run_pipeline.py --skip-transformer

# Just prepare data
python run_pipeline.py --data-only
```

### 4. Launch the active learning app

```bash
streamlit run app/active_learning_app.py
```

### 5. Explore in Jupyter

```bash
jupyter notebook notebooks/analysis.ipynb
```

## Datasets

| Dataset | Source | Size | Purpose |
|---------|--------|------|---------|
| QBias | [GitHub](https://github.com/irgroup/Qbias) | ~21K headlines | Ideology labels (left/center/right) for training |
| Headlines 2007–2022 | [Kaggle](https://www.kaggle.com/datasets/jordankrishnayah/45m-headlines-from-2007-2022-10-largest-sites) | 4.5M headlines | Main analysis corpus |

## Models

### Baseline: TF-IDF + Logistic Regression
- Feature extraction: TF-IDF with bigrams (30K features)
- Classifier: multinomial logistic regression
- Fast to train, decent accuracy (~60-70% on 3-class)

### Transformer: DistilBERT
- Fine-tuned `distilbert-base-uncased` on QBias
- 3 epochs, AdamW optimizer, linear warmup
- Better accuracy (~70-80%) but slower

### Sentiment: VADER
- Rule-based sentiment analyzer tuned for social media / short text
- Compound score: -1 (negative) to +1 (positive)
- Emotionality metric: |compound| (intensity regardless of direction)

## Visualizations

The pipeline generates 7 publication-quality plots:

1. **Ideology Trends** – quarterly ideology scores with 95% CI per publication
2. **Sentiment Trends** – quarterly sentiment with CI
3. **Emotionality Trends** – are headlines getting more emotional over time?
4. **Ideology by Topic** – heatmap of ideology × topic × publication
5. **Sentiment Distribution** – violin plots per publication
6. **Ideology Distribution** – stacked bars by year
7. **Election Effect** – emotionality in election vs. non-election years

## Active Learning App

The Streamlit app implements a crowdsourcing interface for headline emotionality:
- Shows pairs of headlines
- User rates which is more emotional/sensational
- Ratings stored in SQLite
- Active learning prioritizes uncertain/under-rated examples

## Research Questions

1. Do CNN, Fox News, WaPo, and NYT headlines show measurable ideology shifts from 2013–2022?
2. Is there a trend toward more emotionally charged headlines?
3. Do election years correlate with higher emotionality?
4. How do ideology and sentiment patterns differ by topic area?

## License

Academic use. Datasets are subject to their own licenses.
