"""FINAL evaluation of the locked test split. Run it once, deliberately.

This is the second look at the test split: an earlier evaluation exists
(reports/distilbert_v1/test_results.json). Nothing here may be tuned on test; every choice is frozen from
validation or train before test.csv is read:

  * DistilBERT decision threshold 0.291163, baseline TF-IDF + logistic regression (C=30) with its saved log-odds
    threshold (about -1.516), the three-outcome verdict rule and the display-score formula;
  * temperature T, fitted here on validation only (val_probabilities.npy, val_labels.npy);
  * optional exploratory combiner (DistilBERT log-odds + TF-IDF similarity to train fraud/legit/diff; vectoriser
    fit on train only), fitted on validation only, its threshold chosen on out-of-fold validation scores.

Safety checks, in order, all before any test row is read (exit code 2 = refused, 3 = integrity check failed):
  1. --final-evaluation must be given and reports/final_test/final_test_results.json must not exist
     (there is no --overwrite);
  2. SHA-256 of train.csv, val.csv and test.csv must match the split manifest, test.csv and the model weights
     must also match the expected hash fragments, and the baseline's C and threshold must be the frozen ones;
  3. the DistilBERT scorer used here must reproduce the saved validation probabilities.

Reported with group bootstrap 95% intervals by group_id (2000 resamples, seed 42): PR-AUC, ROC-AUC, precision,
recall and F1 at the frozen thresholds, confusion matrices, the verdict-tier table (Clopper-Pearson intervals),
ECE and log-loss before and after temperature, and the paired DistilBERT - baseline PR-AUC difference.
Writes reports/final_test/final_test_results.json and final_test_summary.md.

Usage: python scripts/final_test_eval.py --preflight           # every check and fit, no test rows, no output files
       python scripts/final_test_eval.py --final-evaluation    # the one final run
"""
import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import beta as beta_dist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services import dataset as ds  # noqa: E402
from backend.services.baseline import select_threshold  # noqa: E402
from backend.services.job_analysis import display_risk_score, job_verdict  # noqa: E402

BANNER = ("FINAL TEST EVALUATION. This is the second look at the test split; "
          "an earlier evaluation exists (reports/distilbert_v1/test_results.json).")
EXIT_REFUSED, EXIT_INTEGRITY = 2, 3
SEED = 42
N_BOOT = 2000
TIERS = ("Likely Legitimate", "Needs Verification", "High Fraud Risk")
COMBINER_LABEL = "exploratory, frozen on validation"
CALIBRATION_BINS = 15
ALIGNMENT_ROWS = 32
ALIGNMENT_TOLERANCE = 1e-3   # max |log-odds from this scorer - log-odds from the saved validation probabilities|
SIM_COLS = ["sim_fraud", "sim_legit", "sim_diff"]
BAND_EDGES = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


@dataclass
class Config:
    processed_dir: Path = ROOT / "data" / "processed_v2"
    model_dir: Path = ROOT / "backend" / "models_v2" / "distilbert_job_scam"
    baseline_path: Path = ROOT / "backend" / "models_v2" / "baseline_tfidf_lr.joblib"
    out_dir: Path = ROOT / "reports" / "final_test"
    distil_threshold: float = 0.291163
    expected_baseline_c: float = 30.0
    expected_baseline_threshold: float = -1.516
    baseline_threshold_tolerance: float = 1e-3
    # Hash fragments given for the locked artefacts; the full test/train/val hashes come from the manifest.
    test_hash_fragments: tuple = ("46e46b3e", "2565565082492f")
    weights_hash_fragments: tuple = ("82385cb7", "b9de298f")
    n_boot: int = N_BOOT
    cv_folds: int = 5
    batch_size: int = 16
    extra: dict = field(default_factory=dict)

    @property
    def results_path(self) -> Path:
        return self.out_dir / "final_test_results.json"

    @property
    def summary_path(self) -> Path:
        return self.out_dir / "final_test_summary.md"


class IntegrityError(RuntimeError):
    """A hash, frozen choice or alignment check failed; nothing was scored."""


# --------------------------------------------------------------------------- integrity checks
def _fragments_match(digest: str, fragments: tuple) -> bool:
    return digest.startswith(fragments[0]) and digest.endswith(fragments[1])


def verify_hashes(cfg: Config) -> dict:
    manifest = json.loads((cfg.processed_dir / "split_manifest.json").read_text(encoding="utf-8"))
    hashes = {}
    for name in ("train", "val", "test"):
        entry = manifest["splits"][name]
        actual = ds.file_sha256(cfg.processed_dir / entry["file"])
        if actual != entry["sha256"]:
            raise IntegrityError(f"{name} split does not match the manifest SHA-256")
        hashes[f"{name}_csv_sha256"] = actual
    if not _fragments_match(hashes["test_csv_sha256"], cfg.test_hash_fragments):
        raise IntegrityError("test.csv hash does not match the expected locked hash")
    weights = ds.file_sha256(cfg.model_dir / "model.safetensors")
    if not _fragments_match(weights, cfg.weights_hash_fragments):
        raise IntegrityError("DistilBERT weights hash does not match the expected hash")
    hashes["distilbert_weights_sha256"] = weights
    hashes["baseline_joblib_sha256"] = ds.file_sha256(cfg.baseline_path)
    return hashes


def load_frozen_baseline(cfg: Config) -> dict:
    import joblib

    artifact = joblib.load(cfg.baseline_path)
    pipeline, threshold = artifact["pipeline"], float(artifact["threshold"])
    c_value = float(pipeline.named_steps["lr"].C)
    if c_value != cfg.expected_baseline_c:
        raise IntegrityError(f"baseline C is {c_value:g}, expected {cfg.expected_baseline_c:g}")
    if abs(threshold - cfg.expected_baseline_threshold) > cfg.baseline_threshold_tolerance:
        raise IntegrityError(f"baseline threshold is {threshold:.4f}, expected about {cfg.expected_baseline_threshold}")
    return {"pipeline": pipeline, "threshold": threshold, "C": c_value}


# --------------------------------------------------------------------------- small maths
def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def sigmoid(z) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = CALIBRATION_BINS) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def fit_temperature(val_logodds: np.ndarray, y: np.ndarray) -> dict:
    """Scalar T > 0 minimising validation log-loss of sigmoid(logodds / T)."""
    result = minimize_scalar(lambda t: log_loss(y, sigmoid(val_logodds / t)), bounds=(0.05, 20.0), method="bounded")
    t = float(result.x)
    return {"T": t, "val_log_loss_before": log_loss(y, sigmoid(val_logodds)),
            "val_log_loss_after": log_loss(y, sigmoid(val_logodds / t))}


def confusion(y: np.ndarray, pred: np.ndarray) -> dict:
    return {"tp": int(((pred) & (y == 1)).sum()), "fp": int(((pred) & (y == 0)).sum()),
            "fn": int(((~pred) & (y == 1)).sum()), "tn": int(((~pred) & (y == 0)).sum())}


def prf(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    c = confusion(y, pred)
    precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> list:
    if n == 0:
        return [float("nan"), float("nan")]
    low = 0.0 if k == 0 else float(beta_dist.ppf(alpha / 2, k, n - k + 1))
    high = 1.0 if k == n else float(beta_dist.ppf(1 - alpha / 2, k + 1, n - k))
    return [low, high]


def interval(values, level: float = 95.0) -> list:
    lo, hi = np.percentile(values, [(100 - level) / 2, 100 - (100 - level) / 2])
    return [float(lo), float(hi)]


# --------------------------------------------------------------------------- similarity features / combiner
def fit_similarity(train: pd.DataFrame):
    """TF-IDF vectoriser fit on TRAIN only; returns features(df) = max cosine to train fraud / legit and the difference."""
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=300_000)
    x_train = vec.fit_transform(train["text"].fillna(""))
    fraud = (train[ds.LABEL_COL] == 1).to_numpy()
    fraud_rows, legit_rows = np.flatnonzero(fraud), np.flatnonzero(~fraud)

    def features(df: pd.DataFrame) -> pd.DataFrame:
        x = vec.transform(df["text"].fillna(""))

        def max_sim(rows):
            out = np.empty(x.shape[0])
            for start in range(0, x.shape[0], 500):
                out[start:start + 500] = (x[start:start + 500] @ x_train[rows].T).max(axis=1).toarray().ravel()
            return out

        sf, sl = max_sim(fraud_rows), max_sim(legit_rows)
        return pd.DataFrame({"sim_fraud": sf, "sim_legit": sl, "sim_diff": sf - sl})

    return features


def combiner_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def fit_combiner(val_features: pd.DataFrame, y: np.ndarray, groups: np.ndarray, folds: int) -> dict:
    """Fit on validation only. The decision threshold (max F1) comes from out-of-fold validation scores."""
    oof = np.zeros(len(y))
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
    for train_idx, test_idx in cv.split(val_features, y, groups):
        model = combiner_model().fit(val_features.iloc[train_idx], y[train_idx])
        oof[test_idx] = model.predict_proba(val_features.iloc[test_idx])[:, 1]
    return {"model": combiner_model().fit(val_features, y), "threshold": float(select_threshold(y, oof, "max_f1")),
            "val_oof_pr_auc": float(average_precision_score(y, oof))}


# --------------------------------------------------------------------------- DistilBERT scorer
def make_real_distil_scorer(cfg: Config):
    """Batched log-odds of fraud from the deployed model (logit1 - logit0)."""
    import torch
    from backend.services.distilbert_service import MAX_LENGTH, DistilBERTJobScamService

    service = DistilBERTJobScamService(threshold=cfg.distil_threshold)

    def score(texts) -> np.ndarray:
        texts = [t if t.strip() else " " for t in texts]
        order = np.argsort([len(t) for t in texts])
        out = np.empty(len(texts))
        for start in range(0, len(texts), cfg.batch_size):
            ids = order[start:start + cfg.batch_size]
            batch = service.tokenizer([texts[i] for i in ids], return_tensors="pt", truncation=True,
                                      max_length=MAX_LENGTH, padding=True)
            batch = {k: v.to(service.device) for k, v in batch.items()}
            with torch.inference_mode():
                logits = service.model(**batch).logits
            out[ids] = (logits[:, 1] - logits[:, 0]).cpu().numpy()
        return out

    return score


# --------------------------------------------------------------------------- bootstrap
def group_index(groups: np.ndarray) -> list:
    return list(pd.Series(np.arange(len(groups))).groupby(np.asarray(groups)).indices.values())


def bootstrap(y: np.ndarray, models: dict, calib: dict, groups: np.ndarray, n_boot: int) -> dict:
    """Group bootstrap (resample group_id with replacement). models[name] = {"score": array, "pred": bool array}."""
    by_group = group_index(groups)
    rng = np.random.default_rng(SEED)
    store = {name: {"pr_auc": [], "roc_auc": [], "precision": [], "recall": [], "f1": []} for name in models}
    cal = {k: [] for k in ("ece_before", "ece_after", "log_loss_before", "log_loss_after")}
    diffs, skipped = [], 0
    for _ in range(n_boot):
        chosen = rng.integers(0, len(by_group), len(by_group))
        idx = np.concatenate([by_group[i] for i in chosen])
        yb = y[idx]
        if yb.min() == yb.max():
            skipped += 1
            continue
        for name, m in models.items():
            s, pred = m["score"][idx], m["pred"][idx]
            p, r, f = prf(yb, pred)
            store[name]["pr_auc"].append(average_precision_score(yb, s))
            store[name]["roc_auc"].append(roc_auc_score(yb, s))
            store[name]["precision"].append(p)
            store[name]["recall"].append(r)
            store[name]["f1"].append(f)
        diffs.append(store["distilbert"]["pr_auc"][-1] - store["baseline"]["pr_auc"][-1])
        cal["ece_before"].append(expected_calibration_error(yb, calib["before"][idx]))
        cal["ece_after"].append(expected_calibration_error(yb, calib["after"][idx]))
        cal["log_loss_before"].append(log_loss(yb, calib["before"][idx]))
        cal["log_loss_after"].append(log_loss(yb, calib["after"][idx]))
    return {"models": store, "calibration": cal, "pr_auc_diff": np.array(diffs), "skipped": skipped,
            "used": n_boot - skipped}


# --------------------------------------------------------------------------- evaluation
def _model_block(y, m, threshold, boot_store) -> dict:
    precision, recall, f1 = prf(y, m["pred"])
    point = {"pr_auc": float(average_precision_score(y, m["score"])), "roc_auc": float(roc_auc_score(y, m["score"])),
             "precision": precision, "recall": recall, "f1": f1}
    return {"threshold": float(threshold), **{k: {"value": v, "ci95": interval(boot_store[k])} for k, v in point.items()},
            "confusion": confusion(y, m["pred"])}


def prepare_frozen(cfg: Config, make_distil_scorer) -> dict:
    """
    Every check and every fitted choice that must exist before test rows are read. Reads train.csv and val.csv
    (rows) but test.csv only as bytes for its hash. Shared by --preflight and the final run.
    """
    checks = []
    hashes = verify_hashes(cfg)
    checks.append("train, val and test SHA-256 match the split manifest (test.csv hashed as bytes, no rows read)")
    checks.append("test.csv and the DistilBERT weights match the expected locked hash fragments")
    baseline = load_frozen_baseline(cfg)
    checks.append(f"baseline is C={baseline['C']:g} with the frozen log-odds threshold {baseline['threshold']:.4f}")

    train = pd.read_csv(cfg.processed_dir / "train.csv")
    val = pd.read_csv(cfg.processed_dir / "val.csv")
    val_p = np.load(cfg.model_dir / "val_probabilities.npy")
    y_val = np.load(cfg.model_dir / "val_labels.npy")
    if len(val) != len(val_p) or not (val[ds.LABEL_COL].to_numpy() == y_val).all():
        raise IntegrityError("val.csv does not line up with the saved validation probabilities")
    checks.append("val.csv lines up with the saved validation probabilities and labels")
    val_logodds = logit(val_p)

    distil = make_distil_scorer()
    probe = distil(val["text"].fillna("").astype(str).tolist()[:ALIGNMENT_ROWS])
    gap = float(np.max(np.abs(probe - val_logodds[:ALIGNMENT_ROWS])))
    if gap > ALIGNMENT_TOLERANCE:
        raise IntegrityError(f"DistilBERT scorer does not reproduce saved validation log-odds (max gap {gap:.2e})")
    checks.append(f"DistilBERT scorer reproduces saved validation log-odds on {ALIGNMENT_ROWS} rows (max gap {gap:.1e})")

    # Everything below is fixed from train / validation before test.csv is read.
    temperature = fit_temperature(val_logodds, y_val)
    similarity = fit_similarity(train)
    val_features = pd.concat([pd.DataFrame({"logodds": val_logodds}), similarity(val)], axis=1)
    combiner = fit_combiner(val_features, y_val, val[ds.GROUP_COL].to_numpy(), cfg.cv_folds)
    checks.append("temperature, train-only similarity features and the combiner fitted on validation only")
    return {"checks": checks, "hashes": hashes, "baseline": baseline, "train": train, "val": val, "distil": distil,
            "gap": gap, "temperature": temperature, "similarity": similarity, "combiner": combiner}


def evaluate(cfg: Config, make_distil_scorer) -> dict:
    frozen = prepare_frozen(cfg, make_distil_scorer)
    hashes, baseline, train, val = frozen["hashes"], frozen["baseline"], frozen["train"], frozen["val"]
    distil, gap, temperature = frozen["distil"], frozen["gap"], frozen["temperature"]
    similarity, combiner = frozen["similarity"], frozen["combiner"]

    test = pd.read_csv(cfg.processed_dir / "test.csv")
    seen = set(train[ds.ID_COL]) | set(val[ds.ID_COL]), set(train[ds.GROUP_COL]) | set(val[ds.GROUP_COL])
    if set(test[ds.ID_COL]) & seen[0] or set(test[ds.GROUP_COL]) & seen[1]:
        raise IntegrityError("test overlaps train/validation by job_id or group_id")
    y = test[ds.LABEL_COL].to_numpy(dtype=int)
    texts = test["text"].fillna("").astype(str).tolist()

    test_logodds = distil(texts)
    test_p = sigmoid(test_logodds)
    base_score = baseline["pipeline"].decision_function(texts)
    test_features = pd.concat([pd.DataFrame({"logodds": test_logodds}), similarity(test)], axis=1)
    comb_score = combiner["model"].predict_proba(test_features)[:, 1]

    models = {
        "distilbert": {"score": test_logodds, "pred": test_p >= cfg.distil_threshold},
        "baseline": {"score": base_score, "pred": base_score >= baseline["threshold"]},
        "combiner_exploratory": {"score": comb_score, "pred": comb_score >= combiner["threshold"]},
    }
    calib = {"before": test_p, "after": sigmoid(test_logodds / temperature["T"])}
    boot = bootstrap(y, models, calib, test[ds.GROUP_COL].to_numpy(), cfg.n_boot)

    blocks = {
        "distilbert": _model_block(y, models["distilbert"], cfg.distil_threshold, boot["models"]["distilbert"]),
        "baseline": _model_block(y, models["baseline"], baseline["threshold"], boot["models"]["baseline"]),
        "combiner_exploratory": {"label": COMBINER_LABEL,
                                 **_model_block(y, models["combiner_exploratory"], combiner["threshold"],
                                                boot["models"]["combiner_exploratory"])},
    }
    point_diff = blocks["distilbert"]["pr_auc"]["value"] - blocks["baseline"]["pr_auc"]["value"]

    profiles = test["company_profile"].map(lambda v: ds.clean_field(v) or "")
    verdicts = np.array([job_verdict(float(p), cfg.distil_threshold, pr)[0] for p, pr in zip(test_p, profiles)])
    tiers = []
    for name in TIERS:
        mask = verdicts == name
        n, k = int(mask.sum()), int(y[mask].sum())
        tiers.append({"tier": name, "n": n, "fraud": k, "fraud_rate": k / n if n else None,
                      "fraud_rate_ci95_clopper_pearson": clopper_pearson(k, n)})

    display = np.array([display_risk_score(float(lo), cfg.distil_threshold) for lo in test_logodds])
    bands = []
    for lo, hi in zip(BAND_EDGES, BAND_EDGES[1:]):
        mask = (display >= lo) & (display < hi)
        bands.append({"band": f"{lo}-{hi - 1}", "n": int(mask.sum()), "fraud": int(y[mask].sum())})

    calibration = {
        "temperature": temperature["T"], "temperature_fitted_on": "validation only",
        "val_log_loss_before": temperature["val_log_loss_before"], "val_log_loss_after": temperature["val_log_loss_after"],
        "bins": CALIBRATION_BINS,
        **{k: {"value": v, "ci95": interval(boot["calibration"][k])} for k, v in {
            "ece_before": expected_calibration_error(y, calib["before"]),
            "ece_after": expected_calibration_error(y, calib["after"]),
            "log_loss_before": log_loss(y, calib["before"]), "log_loss_after": log_loss(y, calib["after"])}.items()},
    }
    return {
        "schema_version": 1,
        "banner": BANNER,
        "hashes": hashes,
        "frozen_choices": {
            "distilbert_threshold": cfg.distil_threshold, "baseline_threshold": baseline["threshold"],
            "baseline_C": baseline["C"], "temperature": temperature["T"],
            "combiner": {"label": COMBINER_LABEL, "fitted_on": "validation only",
                         "features": ["logodds", *SIM_COLS], "threshold": combiner["threshold"],
                         "threshold_source": "max-F1 on out-of-fold validation scores",
                         "val_oof_pr_auc": combiner["val_oof_pr_auc"]},
            "distilbert_scorer_alignment_max_gap": gap,
        },
        "data": {"test_rows": int(len(y)), "test_fraud": int(y.sum()), "test_groups": len(group_index(test[ds.GROUP_COL])),
                 "bootstrap": {"resamples_requested": cfg.n_boot, "resamples_used": boot["used"],
                               "resamples_skipped_single_class": boot["skipped"], "seed": SEED, "unit": ds.GROUP_COL}},
        "models": blocks,
        "paired_pr_auc_difference_distilbert_minus_baseline": {
            "value": float(point_diff), "ci95": interval(boot["pr_auc_diff"]),
            "share_of_resamples_above_zero": float(np.mean(boot["pr_auc_diff"] > 0))},
        "verdict_tiers": tiers,
        "calibration": calibration,
        "display_score_bands": bands,
    }


# --------------------------------------------------------------------------- output
def git_info() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=15).stdout.strip()

    try:
        status = run("status", "--porcelain", "--", "scripts/final_test_eval.py")
        state = "untracked" if status.startswith("??") else "modified" if status else "tracked, unmodified"
        return {"head_commit": run("rev-parse", "HEAD") or "unknown", "script_status": state}
    except Exception:
        return {"head_commit": "unknown", "script_status": "unknown"}


def _fmt(block: dict) -> str:
    lo, hi = block["ci95"]
    return f"{block['value']:.3f} ({lo:.3f}-{hi:.3f})"


def summary_markdown(result: dict) -> str:
    lines = [f"# Final test evaluation", "", f"**{BANNER}**", "",
             f"Generated {result['timestamp_utc']} from commit {result['git']['head_commit']} "
             f"(script: {result['git']['script_status']}). Test rows {result['data']['test_rows']}, "
             f"fraud {result['data']['test_fraud']}. 95% intervals: group bootstrap by group_id, "
             f"{result['data']['bootstrap']['resamples_used']} resamples, seed {SEED}.", "",
             "| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | TP / FP / FN / TN |", "|---|---|---|---|---|---|---|"]
    for name, block in result["models"].items():
        label = f"{name} ({block['label']})" if "label" in block else name
        c = block["confusion"]
        lines.append(f"| {label} | {_fmt(block['pr_auc'])} | {_fmt(block['roc_auc'])} | {_fmt(block['precision'])} | "
                     f"{_fmt(block['recall'])} | {_fmt(block['f1'])} | {c['tp']} / {c['fp']} / {c['fn']} / {c['tn']} |")
    d = result["paired_pr_auc_difference_distilbert_minus_baseline"]
    lines += ["", f"Paired PR-AUC difference, DistilBERT minus baseline: {d['value']:+.3f} "
                  f"(95% {d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f}); above zero in {d['share_of_resamples_above_zero']:.1%} of resamples.",
              "", "## Verdict tiers", "", "| Tier | Postings | Fraud | Fraud rate (Clopper-Pearson 95%) |", "|---|---|---|---|"]
    for t in result["verdict_tiers"]:
        lo, hi = t["fraud_rate_ci95_clopper_pearson"]
        rate = "-" if t["fraud_rate"] is None else f"{t['fraud_rate']:.3f} ({lo:.3f}-{hi:.3f})"
        lines.append(f"| {t['tier']} | {t['n']} | {t['fraud']} | {rate} |")
    c = result["calibration"]
    lines += ["", "## Calibration", "", f"Temperature T = {c['temperature']:.3f} (fitted on validation only).", "",
              "| Metric | Before | After |", "|---|---|---|",
              f"| ECE ({c['bins']} bins) | {_fmt(c['ece_before'])} | {_fmt(c['ece_after'])} |",
              f"| Log-loss | {_fmt(c['log_loss_before'])} | {_fmt(c['log_loss_after'])} |", "",
              f"The combiner row is {COMBINER_LABEL}. Nothing was tuned on test. This is a second look at the test split."]
    return "\n".join(lines) + "\n"


def write_outputs(cfg: Config, result: dict) -> None:
    summary = summary_markdown(result)  # built first: a failure here must not leave a results file behind
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    with cfg.results_path.open("x", encoding="utf-8") as f:  # "x": never replaces an existing results file
        json.dump(result, f, indent=2, ensure_ascii=False)
    cfg.summary_path.write_text(summary, encoding="utf-8")


def run_preflight(cfg: Config, make_distil_scorer) -> int:
    """All pre-test checks and fits, no test rows, no output files. Exit 0 if everything passed, 3 otherwise."""
    try:
        frozen = prepare_frozen(cfg, make_distil_scorer)
    except IntegrityError as exc:
        print(f"Preflight FAILED: {exc}.", file=sys.stderr)
        return EXIT_INTEGRITY
    print("Preflight checks passed:")
    for check in frozen["checks"]:
        print(f"  [ok] {check}")
    print("Frozen values:")
    print(f"  DistilBERT threshold   {cfg.distil_threshold}")
    print(f"  temperature T          {frozen['temperature']['T']:.4f}")
    print(f"  baseline threshold     {frozen['baseline']['threshold']:.4f}")
    print(f"  combiner threshold     {frozen['combiner']['threshold']:.4f} ({COMBINER_LABEL})")
    print("Preflight only: test rows were not read and nothing was written. Run with --final-evaluation for the "
          "one final evaluation.")
    return 0


def main(argv=None, cfg: Config | None = None, make_distil_scorer=None) -> int:
    cfg = cfg or Config()
    print("=" * 78)
    print(BANNER)
    print("=" * 78)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--final-evaluation", action="store_true",
                      help="confirm this is the one final evaluation of the locked test split")
    mode.add_argument("--preflight", action="store_true",
                      help="run every pre-test check and fit; never reads test rows and writes nothing")
    args = ap.parse_args(argv)

    if args.preflight:
        return run_preflight(cfg, make_distil_scorer or (lambda: make_real_distil_scorer(cfg)))
    if not args.final_evaluation:
        print("Refusing to run: this reads the locked test split. Pass --final-evaluation only for the final run "
              "(or --preflight to check everything without reading test rows).", file=sys.stderr)
        return EXIT_REFUSED
    if cfg.results_path.exists():
        print(f"Refusing to run: {cfg.results_path.name} already exists. There is no --overwrite.", file=sys.stderr)
        return EXIT_REFUSED

    try:
        result = evaluate(cfg, make_distil_scorer or (lambda: make_real_distil_scorer(cfg)))
    except IntegrityError as exc:
        print(f"Stopped before scoring: {exc}.", file=sys.stderr)
        return EXIT_INTEGRITY
    result["timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result["git"] = git_info()
    write_outputs(cfg, result)
    print(f"Wrote {cfg.results_path.name} and {cfg.summary_path.name} to {cfg.out_dir.name}/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
