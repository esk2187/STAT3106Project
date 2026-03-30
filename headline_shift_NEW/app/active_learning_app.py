"""
Active Learning Crowdsourcing Interface for headline emotionality rating.

Backed by Supabase (persistent cloud PostgreSQL) — works on Streamlit Community
Cloud and locally. Credentials come from .streamlit/secrets.toml.

Run locally:
    streamlit run app/active_learning_app.py

Deploy:
    Push to GitHub → connect repo on share.streamlit.io → add secrets in dashboard.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Database (Supabase) ─────────────────────────────────────────────────────

@st.cache_resource
def get_db():
    """Return a cached Supabase client. Reads URL + key from st.secrets."""
    from supabase import create_client
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])


def _get_score_row(db, headline: str) -> dict:
    """Fetch current win/loss/tie counts for a headline, or return zeroes."""
    result = db.table("headline_scores").select("*").eq("headline", headline).execute()
    if result.data:
        return result.data[0]
    return {"headline": headline, "wins": 0, "losses": 0, "ties": 0,
            "comparisons": 0, "uncertainty": 1.0}


def record_comparison(headline_a: str, headline_b: str, choice: str):
    """Save a pairwise comparison and update per-headline win/loss counts."""
    db = get_db()

    # Insert the raw comparison
    db.table("comparisons").insert({
        "headline_a": headline_a,
        "headline_b": headline_b,
        "choice": choice,
    }).execute()

    # Update scores for each headline (read-then-write is fine at this scale)
    for h, is_winner, is_loser in [
        (headline_a, choice == "A", choice == "B"),
        (headline_b, choice == "B", choice == "A"),
    ]:
        row = _get_score_row(db, h)
        wins  = row["wins"]  + (1 if is_winner else 0)
        losses = row["losses"] + (1 if is_loser  else 0)
        ties  = row["ties"]  + (1 if choice == "equal" else 0)
        comps = row["comparisons"] + 1
        uncertainty = (1.0 / (1.0 + comps)) + (abs(wins / comps - 0.5) if comps > 0 else 0.5)

        db.table("headline_scores").upsert({
            "headline": h, "wins": wins, "losses": losses,
            "ties": ties, "comparisons": comps, "uncertainty": uncertainty,
        }).execute()


def get_pair_active_learning(headlines: list) -> tuple:
    """
    Select a pair prioritising uncertain / under-compared headlines.
    Active learning: pair unseen headlines first, then most uncertain seen ones.
    """
    db = get_db()
    result = db.table("headline_scores").select("*").execute()
    scores_df = pd.DataFrame(result.data) if result.data else pd.DataFrame()
    scored = set(scores_df["headline"].tolist()) if len(scores_df) > 0 else set()

    unseen = [h for h in headlines if h not in scored]

    if len(unseen) >= 2:
        return tuple(random.sample(unseen, 2))

    if len(unseen) == 1:
        a = unseen[0]
        b = (random.choice(scores_df.nlargest(5, "uncertainty")["headline"].tolist())
             if len(scores_df) > 0
             else random.choice([h for h in headlines if h != a]))
        return a, b

    if len(scores_df) >= 2:
        top = scores_df.nlargest(min(20, len(scores_df)), "uncertainty")
        sample = top.sample(min(2, len(top)))["headline"].tolist()
        if len(sample) == 2:
            return sample[0], sample[1]

    pair = random.sample(headlines, min(2, len(headlines)))
    return (pair[0], pair[1]) if len(pair) == 2 else (pair[0], pair[0])


def get_stats():
    """Return (total_comparisons, top_headlines_df)."""
    db = get_db()
    try:
        comps = db.table("comparisons").select("*", count="exact").execute()
        total = comps.count or 0
        top_res = (db.table("headline_scores")
                     .select("*")
                     .order("wins", desc=True)
                     .limit(10)
                     .execute())
        top = pd.DataFrame(top_res.data) if top_res.data else pd.DataFrame()
    except Exception:
        total = 0
        top = pd.DataFrame()
    return total, top


# ── Bradley-Terry model ─────────────────────────────────────────────────────

def compute_bradley_terry_scores(max_iter=500, tol=1e-6) -> dict:
    """
    Fit a Bradley-Terry model to the stored pairwise comparisons.

    Assigns a continuous strength score β_i to each headline such that
    P(i beats j) = β_i / (β_i + β_j).  Iterative MLE update:

        β_i  ←  W_i / Σ_{j≠i} [ n_ij / (β_i + β_j) ]

    Ties count as 0.5 wins each.
    Returns {headline: score} normalised to [0, 1], or {} if insufficient data.
    """
    db = get_db()
    try:
        result = db.table("comparisons").select("headline_a, headline_b, choice").execute()
        if not result.data:
            return {}
        comparisons = pd.DataFrame(result.data)
    except Exception:
        return {}

    headlines = list(set(comparisons["headline_a"].tolist() + comparisons["headline_b"].tolist()))
    n = len(headlines)
    if n < 2:
        return {}

    idx = {h: i for i, h in enumerate(headlines)}
    wins = np.zeros(n)
    n_comparisons = np.zeros((n, n))

    for _, row in comparisons.iterrows():
        i, j = idx[row["headline_a"]], idx[row["headline_b"]]
        n_comparisons[i, j] += 1
        n_comparisons[j, i] += 1
        if row["choice"] == "A":
            wins[i] += 1
        elif row["choice"] == "B":
            wins[j] += 1
        else:
            wins[i] += 0.5
            wins[j] += 0.5

    beta = np.ones(n)
    for _ in range(max_iter):
        beta_old = beta.copy()
        for i in range(n):
            denom = sum(
                n_comparisons[i, j] / (beta[i] + beta[j])
                for j in range(n) if j != i and n_comparisons[i, j] > 0
            )
            if denom > 0:
                beta[i] = wins[i] / denom
        total = beta.sum()
        if total > 0:
            beta /= total / n
        if np.max(np.abs(beta - beta_old)) < tol:
            break

    beta_min, beta_max = beta.min(), beta.max()
    beta_norm = (
        (beta - beta_min) / (beta_max - beta_min)
        if beta_max > beta_min
        else np.full(n, 0.5)
    )
    return {headlines[i]: float(beta_norm[i]) for i in range(n)}


def build_emotionality_csv():
    """
    Compute Bradley-Terry scores and return a CSV as bytes for st.download_button.
    Returns None if there is not enough data yet.
    """
    scores = compute_bradley_terry_scores()
    if not scores:
        return None

    db = get_db()
    result = db.table("headline_scores").select("headline, comparisons").execute()
    counts = (
        {row["headline"]: row["comparisons"] for row in result.data}
        if result.data else {}
    )

    rows = [
        {"headline": h, "emotionality_score": s, "num_comparisons": counts.get(h, 0)}
        for h, s in scores.items()
    ]
    df = pd.DataFrame(rows).sort_values("emotionality_score", ascending=False)
    return df.to_csv(index=False).encode("utf-8")


# ── Headline loader ─────────────────────────────────────────────────────────

@st.cache_data
def load_headlines() -> list:
    """
    Load headlines for annotation.
    Tries the full filtered corpus first (available in Colab), then falls back
    to the QBias data that ships with the repo (always available).
    """
    base = os.path.dirname(__file__)
    candidates = [
        os.path.join(base, "..", "data", "processed", "headlines_filtered.csv"),
        os.path.join(base, "..", "data", "qbias", "allsides_balanced_news_headlines-texts.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "headline" in df.columns:
                headlines = df["headline"].dropna().unique().tolist()
                random.shuffle(headlines)
                return headlines
    return []


# ── Streamlit UI ────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Headline Emotionality Rater", layout="wide")

    st.title("🗞️ Headline Emotionality Rater")
    st.markdown("""
    **Help train our model!** Compare pairs of headlines and select which one
    feels more *emotional* or *sensational*. Your ratings improve our classifier.

    The system uses **active learning** to prioritize the most informative comparisons.
    """)

    all_headlines = load_headlines()
    if len(all_headlines) < 2:
        st.error("No headlines found. Check that data files are in place and secrets are configured.")
        st.stop()

    # Session state: keep a pair until the user acts on it
    if "pair" not in st.session_state or st.session_state.get("refresh", False):
        a, b = get_pair_active_learning(all_headlines)
        st.session_state["pair"] = (a, b)
        st.session_state["refresh"] = False

    headline_a, headline_b = st.session_state["pair"]

    # Main comparison UI
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📰 Headline A")
        st.markdown(f"### {headline_a}")
    with col2:
        st.subheader("📰 Headline B")
        st.markdown(f"### {headline_b}")

    st.markdown("---")
    st.markdown("### Which headline is more **emotional / sensational**?")

    btn1, btn2, btn3 = st.columns(3)
    with btn1:
        if st.button("⬅️ Headline A is more emotional", use_container_width=True):
            record_comparison(headline_a, headline_b, "A")
            st.session_state["refresh"] = True
            st.rerun()
    with btn2:
        if st.button("🤝 About equal", use_container_width=True):
            record_comparison(headline_a, headline_b, "equal")
            st.session_state["refresh"] = True
            st.rerun()
    with btn3:
        if st.button("➡️ Headline B is more emotional", use_container_width=True):
            record_comparison(headline_a, headline_b, "B")
            st.session_state["refresh"] = True
            st.rerun()

    if st.button("⏭️ Skip this pair"):
        st.session_state["refresh"] = True
        st.rerun()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.header("📊 Annotation Stats")
    total, top = get_stats()
    st.sidebar.metric("Total Comparisons", total)

    MIN_ANNOTATIONS = 20
    if total < MIN_ANNOTATIONS:
        st.sidebar.progress(min(total / MIN_ANNOTATIONS, 1.0))
        st.sidebar.caption(f"{total} / {MIN_ANNOTATIONS} comparisons before export is recommended")

    # Show Bradley-Terry scores once there's enough data
    bt_scores = compute_bradley_terry_scores()
    if bt_scores:
        st.sidebar.markdown("### Top Emotional Headlines (Bradley-Terry)")
        bt_df = (
            pd.DataFrame(list(bt_scores.items()), columns=["headline", "bt_score"])
            .sort_values("bt_score", ascending=False)
            .head(10)
        )
        for _, row in bt_df.iterrows():
            st.sidebar.markdown(f"- **{row['bt_score']:.2f}**: {row['headline'][:60]}…")
    elif len(top) > 0:
        st.sidebar.markdown("### Top Emotional Headlines (win rate)")
        for _, row in top.iterrows():
            win_rate = row["wins"] / max(row["comparisons"], 1) * 100
            st.sidebar.markdown(f"- **{win_rate:.0f}%** ({row['comparisons']} comps): {row['headline'][:60]}…")

    # Export
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Export for Training")
    if total >= 2:
        csv_bytes = build_emotionality_csv()
        if csv_bytes:
            st.sidebar.download_button(
                label="⬇️ Download Labels CSV",
                data=csv_bytes,
                file_name="emotionality_labels.csv",
                mime="text/csv",
            )
            st.sidebar.caption(
                "Download this file, upload it to Colab, then run:\n"
                "`--emotionality-labels emotionality_labels.csv`"
            )
    else:
        st.sidebar.caption("Make at least 2 comparisons to unlock export.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### How it works")
    st.sidebar.markdown("""
    1. We show you two headlines
    2. You pick which feels more emotional
    3. **Active learning** picks the next pair based on what's most informative
    4. Scores are computed via the **Bradley-Terry** model
    5. Download labels to train the dual-head DistilBERT
    """)


if __name__ == "__main__":
    main()
