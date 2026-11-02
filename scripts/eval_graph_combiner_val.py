"""Exploratory combiner evaluation on VALIDATION ONLY. Writes only the gitignored graph cache.

Does adding TF-IDF similarity-to-train and entity-graph features to DistilBERT's log-odds
improve fraud-class PR-AUC? Everything external is fit on train only (DistilBERT, TF-IDF
vectoriser, entity graph); the combiner itself is fit and scored by repeated group-aware
cross-validation on val.csv. Do not point this at the test split.

Usage: python scripts/eval_graph_combiner_val.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services import dataset as ds  # noqa: E402
from backend.services import entity_graph as eg  # noqa: E402

# --- Success bars: fixed before the first run, do not edit after seeing results ---------------
MIN_MEAN_PR_AUC_GAIN = 0.03        # mean per-fold PR-AUC gain over raw DistilBERT log-odds
INTERVAL_PERCENTILES = (2.5, 97.5)  # 95% interval of the paired fold differences; lower end must be > 0
PRIMARY_ARM = "all"                 # the one arm the adoption decision is made on
N_REPEATS = 10
N_FOLDS = 5
SEED = 42
# -----------------------------------------------------------------------------------------------

PROCESSED = ROOT / "data" / "processed_v2"
MD = ROOT / "backend" / "models_v2" / "distilbert_job_scam"

SIM_COLS = ["sim_fraud", "sim_legit", "sim_diff"]
GRAPH_COLS = ["ents_n", "ents_fraud", "ents_rate", "emailphone_n", "emailphone_fraud"]
ARMS = {
    "logistic": ["logodds"],
    "sim": ["logodds", *SIM_COLS],
    "graph": ["logodds", *GRAPH_COLS],   # graph-only ablation: entity features, no similarity
    "all": ["logodds", *SIM_COLS, *GRAPH_COLS],
    "graph_alone": GRAPH_COLS,           # diagnostic: entity features without DistilBERT
}


def similarity_features(train: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    """Max cosine similarity of each val posting to train fraud / train legit postings."""
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=300_000)
    x_train = vec.fit_transform(train["text"].fillna(""))
    x_val = vec.transform(val["text"].fillna(""))
    fraud = (train[ds.LABEL_COL] == 1).to_numpy()

    def max_sim(rows):
        out = np.empty(x_val.shape[0])
        for start in range(0, x_val.shape[0], 500):
            out[start:start + 500] = (x_val[start:start + 500] @ x_train[rows].T).max(axis=1).toarray().ravel()
        return out

    sim_fraud, sim_legit = max_sim(np.where(fraud)[0]), max_sim(np.where(~fraud)[0])
    return pd.DataFrame({"sim_fraud": sim_fraud, "sim_legit": sim_legit, "sim_diff": sim_fraud - sim_legit})


def main():
    print("=" * 78)
    print("VALIDATION ONLY - EXPLORATORY. Not a test-set result; the combiner is not wired into the app.")
    print("=" * 78)

    train = pd.read_csv(PROCESSED / "train.csv")
    val = pd.read_csv(PROCESSED / "val.csv")
    p = np.load(MD / "val_probabilities.npy")
    y = np.load(MD / "val_labels.npy")
    assert len(val) == len(p) and (val[ds.LABEL_COL].to_numpy() == y).all()
    assert not set(train[ds.ID_COL]) & set(val[ds.ID_COL]) and not set(train[ds.GROUP_COL]) & set(val[ds.GROUP_COL])

    graph = eg.build_or_load(PROCESSED)
    n_entities = sum(1 for _, d in graph.nodes(data=True) if d["bipartite"] == 1)
    print(f"entity graph (train only): {graph.graph['n_train']} postings, {n_entities} entities, "
          f"{graph.graph['hubs_removed']} hub entities removed (> {graph.graph['hub_max_postings']} postings)")

    pc = np.clip(p, 1e-12, 1 - 1e-12)
    feats = pd.DataFrame({"logodds": np.log(pc / (1 - pc))})
    feats = pd.concat([feats, similarity_features(train, val)], axis=1)
    graph_rows = [eg.graph_features(graph, row) for row in val.to_dict("records")]
    feats = pd.concat([feats, pd.DataFrame(graph_rows)], axis=1)
    for col in ("ents_n", "ents_fraud", "emailphone_n", "emailphone_fraud"):
        feats[col] = np.log1p(feats[col])

    linked_fraud = np.array([r["ents_fraud"] > 0 for r in graph_rows])
    print(f"\nentity coverage: val rows with >= 1 fraud-linked entity: {int(linked_fraud.sum())}/{len(val)} "
          f"({linked_fraud.mean():.1%}); frauds covered: {int(linked_fraud[y == 1].sum())}/{int(y.sum())}; "
          f"legit rows with a fraud-linked entity: {int(linked_fraud[y == 0].sum())}")
    print(f"rows with any linked train posting: {int(sum(r['ents_n'] > 0 for r in graph_rows))}/{len(val)}")

    groups = val[ds.GROUP_COL].to_numpy()
    scores = {arm: [] for arm in ["raw", *ARMS]}
    for repeat in range(N_REPEATS):
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + repeat)
        for train_idx, test_idx in cv.split(feats, y, groups):
            scores["raw"].append(average_precision_score(y[test_idx], feats["logodds"].to_numpy()[test_idx]))
            for arm, cols in ARMS.items():
                model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
                model.fit(feats.loc[train_idx, cols], y[train_idx])
                prob = model.predict_proba(feats.loc[test_idx, cols])[:, 1]
                scores[arm].append(average_precision_score(y[test_idx], prob))
    scores = {arm: np.array(v) for arm, v in scores.items()}

    print(f"\n{N_REPEATS} repeats x StratifiedGroupKFold({N_FOLDS}) by group_id = {len(scores['raw'])} folds")
    print(f"{'arm':<14}{'mean PR-AUC':>12}{'sd':>8}   paired diff vs raw: mean  [{INTERVAL_PERCENTILES[0]}%, "
          f"{INTERVAL_PERCENTILES[1]}%]   folds better")
    print(f"{'raw':<14}{scores['raw'].mean():>12.4f}{scores['raw'].std(ddof=1):>8.4f}")
    for arm in ARMS:
        diff = scores[arm] - scores["raw"]
        lo, hi = np.percentile(diff, INTERVAL_PERCENTILES)
        print(f"{arm:<14}{scores[arm].mean():>12.4f}{scores[arm].std(ddof=1):>8.4f}   "
              f"{diff.mean():>+8.4f}  [{lo:+.4f}, {hi:+.4f}]   {int((diff > 0).sum())}/{len(diff)}")

    print("\nincremental: paired diff vs logistic-on-log-odds")
    for arm in ("sim", "graph", "all"):
        diff = scores[arm] - scores["logistic"]
        lo, hi = np.percentile(diff, INTERVAL_PERCENTILES)
        print(f"  {arm:<12}{diff.mean():>+8.4f}  [{lo:+.4f}, {hi:+.4f}]")

    diff = scores[PRIMARY_ARM] - scores["raw"]
    lo, _ = np.percentile(diff, INTERVAL_PERCENTILES)
    adopt = diff.mean() >= MIN_MEAN_PR_AUC_GAIN and lo > 0
    print(f"\nDECISION (pre-registered, arm '{PRIMARY_ARM}'): mean gain {diff.mean():+.4f} "
          f"(bar >= {MIN_MEAN_PR_AUC_GAIN}), interval lower end {lo:+.4f} (bar > 0) -> "
          f"{'ADOPT' if adopt else 'DO NOT ADOPT'}")
    print("Folds share validation rows across repeats, so the interval describes fold-to-fold spread, "
          "not an independent-sample confidence interval. VALIDATION ONLY - EXPLORATORY.")


if __name__ == "__main__":
    main()
