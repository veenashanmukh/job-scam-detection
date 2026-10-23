"""Baseline: word 1-2gram TF-IDF + Logistic Regression.

Rules enforced here (see README):
  * Only the train and val splits are ever loaded. The locked test split is never touched.
  * The TF-IDF vocabulary/IDF are fit on train only.
  * C is chosen on validation average precision (PR-AUC); the decision threshold is chosen on validation by a rule
    fixed in advance (default max-F1, or max recall subject to a precision floor), then frozen.
  * The run record contains hashes, settings and metrics, never posting text.
"""
from __future__ import annotations

import csv
import json
import platform
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline

from backend.services import dataset as ds

DEFAULT_C_GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
THRESHOLD_RULES = ("max_f1", "recall_at_precision")
MODEL_FILE = "baseline_tfidf_lr.joblib"
RESULTS_COLUMNS = ["person", "experiment", "model", "metric", "value", "date"]


# --------------------------------------------------------------------------- data / model
def xy(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Model text (empty-text rows round-trip through CSV as NaN) and integer labels."""
    return df["text"].fillna("").astype(str).tolist(), df[ds.LABEL_COL].to_numpy(dtype=int)


def build_pipeline(C: float, seed: int = ds.SEED) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=300_000)),
        ("lr", LogisticRegression(C=C, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=seed)),
    ])


# --------------------------------------------------------------------------- metrics
def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def metrics_at_threshold(y: np.ndarray, scores: np.ndarray, thr: float) -> dict:
    """Fraud-class (label 1) metrics when predicting fraud for score >= thr."""
    pred = scores >= thr
    tp = int(((pred) & (y == 1)).sum())
    fp = int(((pred) & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum())
    tn = int(((~pred) & (y == 0)).sum())
    p, r = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
    return {"threshold": float(thr), "precision": p, "recall": r, "f1": _safe_div(2 * p * r, p + r),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_fraud": int((y == 1).sum()), "n": int(len(y))}


def ranking_metrics(y: np.ndarray, scores: np.ndarray) -> dict:
    two_classes = len(np.unique(y)) == 2
    return {"average_precision": float(average_precision_score(y, scores)) if two_classes else float("nan"),
            "roc_auc": float(roc_auc_score(y, scores)) if two_classes else float("nan")}


def select_threshold(y: np.ndarray, scores: np.ndarray, rule: str = "max_f1",
                     min_precision: float | None = None) -> float:
    """Pick the decision threshold on VALIDATION scores only. Ties go to the higher (more conservative) threshold."""
    if rule not in THRESHOLD_RULES:
        raise ValueError(f"rule must be one of {THRESHOLD_RULES}")
    precision, recall, thr = precision_recall_curve(y, scores)
    p, r = precision[:-1], recall[:-1]  # the last PR point has no threshold
    if len(thr) == 0:
        raise ValueError("cannot select a threshold from constant/empty scores")
    if rule == "max_f1":
        f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
        idx = np.flatnonzero(f1 >= f1.max() - 1e-12)[-1]
    else:
        if min_precision is None:
            raise ValueError("min_precision is required for rule 'recall_at_precision'")
        feasible = p >= min_precision
        if not feasible.any():
            raise ValueError(f"no threshold reaches precision >= {min_precision} on validation")
        best_recall = r[feasible].max()
        idx = np.flatnonzero(feasible & (r >= best_recall - 1e-12))[-1]
    return float(thr[idx])


def bootstrap_ci(y: np.ndarray, scores: np.ndarray, thr: float, n_boot: int = 1000, seed: int = ds.SEED) -> dict:
    """Stratified bootstrap (positives and negatives resampled separately) 95% percentile intervals."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rows = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        m = metrics_at_threshold(y[idx], scores[idx], thr)
        rows.append((m["precision"], m["recall"], m["f1"], average_precision_score(y[idx], scores[idx])))
    arr = np.array(rows)
    names = ("precision", "recall", "f1", "average_precision")
    return {n: [float(np.percentile(arr[:, i], 2.5)), float(np.percentile(arr[:, i], 97.5))]
            for i, n in enumerate(names)}


def select_C(Xtr, ytr, Xva, yva, grid=DEFAULT_C_GRID, seed: int = ds.SEED):
    """Fit on train for each C, score on validation by average precision. Ties -> smaller C."""
    table, fitted = [], {}
    for C in grid:
        pipe = build_pipeline(C, seed).fit(Xtr, ytr)
        s = pipe.decision_function(Xva)
        table.append({"C": float(C), "val_average_precision": ranking_metrics(yva, s)["average_precision"]})
        fitted[float(C)] = pipe
    best = sorted(table, key=lambda t: (-round(t["val_average_precision"], 12), t["C"]))[0]["C"]
    return best, table, fitted[best]


# --------------------------------------------------------------------------- run
def run_baseline(processed_dir, out_dir, model_dir, c_grid=DEFAULT_C_GRID, rule: str = "max_f1",
                 min_precision: float | None = None, seed: int = ds.SEED, n_boot: int = 1000) -> dict:
    import joblib

    processed_dir, out_dir, model_dir = Path(processed_dir), Path(out_dir), Path(model_dir)
    mpath = processed_dir / "split_manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    if manifest.get("leakage", {}).get("all_passed") is not True:
        raise RuntimeError("split_manifest.json does not record passed leakage checks; refusing to train.")
    if manifest.get("near_duplicate_threshold") != ds.DEFAULT_NEAR_THRESHOLD:
        raise RuntimeError("split was not made with the 0.90 near-duplicate threshold; refusing to train.")

    train = ds.load_split("train", processed_dir)   # hash-verified against the manifest
    val = ds.load_split("val", processed_dir)       # NOTE: the test split is deliberately never loaded
    Xtr, ytr = xy(train)
    Xva, yva = xy(val)

    best_C, table, pipe = select_C(Xtr, ytr, Xva, yva, c_grid, seed)
    s_tr, s_va = pipe.decision_function(Xtr), pipe.decision_function(Xva)
    thr = select_threshold(yva, s_va, rule, min_precision)

    val_metrics = {**metrics_at_threshold(yva, s_va, thr), **ranking_metrics(yva, s_va)}
    record = {
        "model": "tfidf_lr", "evaluated_split": "val", "test_split_loaded": False,
        "seed": seed, "n_bootstrap": n_boot,
        "split_manifest_sha256": ds.file_sha256(mpath),
        "split_file_sha256": {k: v["sha256"] for k, v in manifest["splits"].items()},
        "pipeline_code_sha256_in_manifest": manifest.get("pipeline_code_sha256"),
        "baseline_code_sha256": ds.file_sha256(__file__),
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "features": "word 1-2gram TF-IDF, sublinear_tf, min_df=2, max_features=300000; fit on train only",
        "classifier": "LogisticRegression(class_weight='balanced', solver='liblinear')",
        "vocab_size": int(len(pipe.named_steps["tfidf"].vocabulary_)),
        "c_grid_results": table, "chosen_C": best_C,
        "threshold_rule": rule, "min_precision": min_precision, "frozen_threshold": thr,
        "threshold_note": "applied to LogisticRegression decision_function (log-odds); predict fraud if score >= threshold",
        "val": val_metrics, "val_bootstrap_95ci": bootstrap_ci(yva, s_va, thr, n_boot, seed),
        "train_at_same_threshold": {**metrics_at_threshold(ytr, s_tr, thr), **ranking_metrics(ytr, s_tr)},
        "split_sizes": {"train": int(len(train)), "val": int(len(val))},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline_run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    joblib.dump({"pipeline": pipe, "threshold": thr, "record_sha_hint": record["split_manifest_sha256"]},
                model_dir / MODEL_FILE)
    return record


def top_features(pipe: Pipeline, k: int = 15) -> dict[str, list[str]]:
    names = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    coef = pipe.named_steps["lr"].coef_[0]
    order = np.argsort(coef)
    return {"fraud": names[order[::-1][:k]].tolist(), "legitimate": names[order[:k]].tolist()}


def append_results(csv_path, person: str, experiment: str, model: str, metrics: dict[str, float]) -> None:
    """Append rows to the shared data/evaluation/results.csv (person,experiment,model,metric,value,date)."""
    csv_path = Path(csv_path)
    new = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(RESULTS_COLUMNS)
        for k, v in metrics.items():
            w.writerow([person, experiment, model, k, round(float(v), 6), date.today().isoformat()])
