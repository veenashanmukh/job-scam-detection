"""Does the way a posting is typed into the app change the DistilBERT score? VALIDATION ONLY, offline, no LLM.

Sample of val.csv: every fraud posting plus --n-legit (default 400, at most 400) random legitimate postings, seed 42.
Each posting is scored in three input formats with the deployed model:
  structured        build_text of the five fields, as the form sends it ("Title: ...", "Description: ...");
  plain             the same cleaned fields joined with newlines and no labels, as the paste box sends it
                    (the paste path posts the text exactly as typed);
  description-only  build_text with the whole plain text placed in the Description field.
Reported per format: PR-AUC, recall and false positives at the deployed threshold, plus the group bootstrap
(group_id, seed 42, 2000 resamples) 95% interval of the PR-AUC difference versus structured.

The sample is enriched for fraud (all frauds), so its PR-AUC is not comparable with full-validation numbers; the
comparison between formats is what matters. Never opens test.csv.

Usage: python scripts/eval_input_formats_val.py [--n-legit 400] [--n-boot 2000]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services import dataset as ds  # noqa: E402

BANNER = "VALIDATION ONLY - INPUT FORMAT CHECK. Not a test-set result; the sample contains every fraud posting."
SEED = 42
MAX_LEGIT = 400
N_BOOT = 2000
THRESHOLD = 0.291163
ALIGNMENT_TOLERANCE = 1e-3
FORMATS = ("structured", "plain", "description-only")
VAL_CSV = ROOT / "data" / "processed_v2" / "val.csv"
VAL_PROBABILITIES = ROOT / "backend" / "models_v2" / "distilbert_job_scam" / "val_probabilities.npy"


def sample_rows(val: pd.DataFrame, n_legit: int) -> pd.DataFrame:
    """All fraud rows plus n_legit random legitimate rows (seed 42), in the original row order."""
    rng = np.random.default_rng(SEED)
    fraud = val.index[val[ds.LABEL_COL] == 1].to_numpy()
    legit = val.index[val[ds.LABEL_COL] == 0].to_numpy()
    chosen = np.concatenate([fraud, rng.choice(legit, min(n_legit, len(legit)), replace=False)])
    return val.loc[np.sort(chosen)]


def format_texts(row: dict) -> dict:
    cleaned = {field: ds.clean_field(row.get(field)) for field in ds.TEXT_FIELDS}
    plain = "\n".join(value for value in cleaned.values() if value)
    return {
        "structured": ds.build_text(pd.Series(cleaned)),
        "plain": plain,
        "description-only": ds.build_text(pd.Series({"description": plain})),
    }


def group_index(groups: np.ndarray) -> list:
    return list(pd.Series(np.arange(len(groups))).groupby(np.asarray(groups)).indices.values())


def format_metrics(y: np.ndarray, logodds: np.ndarray, threshold: float = THRESHOLD) -> dict:
    flagged = 1.0 / (1.0 + np.exp(-logodds)) >= threshold
    tp, fp = int((flagged & (y == 1)).sum()), int((flagged & (y == 0)).sum())
    return {"pr_auc": float(average_precision_score(y, logodds)), "recall": tp / int((y == 1).sum()),
            "tp": tp, "false_positives": fp, "precision": tp / (tp + fp) if tp + fp else 0.0,
            "n_flagged": tp + fp}


def bootstrap_pr_auc_difference(y: np.ndarray, base: np.ndarray, other: np.ndarray, groups: np.ndarray,
                                n_boot: int = N_BOOT) -> dict:
    """PR-AUC(other) - PR-AUC(base) over group resamples; 95% percentile interval."""
    by_group = group_index(groups)
    rng = np.random.default_rng(SEED)
    diffs, skipped = [], 0
    for _ in range(n_boot):
        idx = np.concatenate([by_group[i] for i in rng.integers(0, len(by_group), len(by_group))])
        if y[idx].min() == y[idx].max():
            skipped += 1
            continue
        diffs.append(average_precision_score(y[idx], other[idx]) - average_precision_score(y[idx], base[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"difference": float(average_precision_score(y, other) - average_precision_score(y, base)),
            "ci95": [float(lo), float(hi)], "resamples_used": len(diffs), "skipped": skipped}


def evaluate(val: pd.DataFrame, probabilities: np.ndarray, score, n_legit: int, n_boot: int) -> dict:
    """score(texts) -> log-odds array. probabilities are the saved validation probabilities aligned with val."""
    if len(val) != len(probabilities):
        raise ValueError("val.csv and val_probabilities.npy have different lengths")
    sample = sample_rows(val, n_legit)
    y = sample[ds.LABEL_COL].to_numpy(dtype=int)
    texts = {name: [] for name in FORMATS}
    for row in sample.to_dict("records"):
        for name, text in format_texts(row).items():
            texts[name].append(text)
    scores = {name: np.asarray(score(texts[name]), dtype=float) for name in FORMATS}

    saved = np.clip(np.asarray(probabilities)[val.index.get_indexer(sample.index)], 1e-12, 1 - 1e-12)
    gap = float(np.max(np.abs(scores["structured"] - np.log(saved / (1 - saved)))))
    if gap > ALIGNMENT_TOLERANCE:
        raise RuntimeError(f"structured scores do not reproduce the saved validation log-odds (max gap {gap:.2e})")

    groups = sample[ds.GROUP_COL].to_numpy()
    result = {"n": len(y), "n_fraud": int(y.sum()), "n_legit": int((y == 0).sum()), "alignment_gap": gap,
              "formats": {}, "mean_logodds_shift_vs_structured": {}}
    for name in FORMATS:
        result["formats"][name] = format_metrics(y, scores[name])
        if name != "structured":
            result["formats"][name]["pr_auc_difference_vs_structured"] = bootstrap_pr_auc_difference(
                y, scores["structured"], scores[name], groups, n_boot)
            result["mean_logodds_shift_vs_structured"][name] = float(np.mean(scores[name] - scores["structured"]))
    return result


def report_lines(result: dict) -> list[str]:
    lines = [BANNER, f"{result['n']} postings: {result['n_fraud']} fraud (all) + {result['n_legit']} legitimate "
                     f"(random, seed {SEED}); threshold {THRESHOLD}",
             f"structured scores reproduce the saved validation log-odds (max gap {result['alignment_gap']:.1e})", "",
             f"{'format':<18}{'PR-AUC':>8}{'recall':>8}{'false pos':>11}{'precision':>11}   PR-AUC difference vs structured"]
    for name in FORMATS:
        m = result["formats"][name]
        diff = m.get("pr_auc_difference_vs_structured")
        tail = (f"{diff['difference']:+.4f}  (95% {diff['ci95'][0]:+.4f} to {diff['ci95'][1]:+.4f}, "
                f"{diff['resamples_used']} group resamples)") if diff else "-"
        lines.append(f"{name:<18}{m['pr_auc']:>8.4f}{m['recall']:>8.3f}{m['false_positives']:>11}{m['precision']:>11.3f}   {tail}")
    lines += ["", "Mean log-odds shift vs structured: " +
              ", ".join(f"{k} {v:+.3f}" for k, v in result["mean_logodds_shift_vs_structured"].items()),
              f"Legitimate postings in the sample: {result['n_legit']}; 'false pos' counts those flagged at the threshold.",
              "", BANNER]
    return lines


def make_real_scorer(batch_size: int = 16):
    """Batched log-odds of fraud from the deployed model (logit1 - logit0)."""
    import torch
    from backend.services.distilbert_service import MAX_LENGTH, DistilBERTJobScamService

    service = DistilBERTJobScamService(threshold=THRESHOLD)

    def score(texts) -> np.ndarray:
        texts = [t if t.strip() else " " for t in texts]
        order = np.argsort([len(t) for t in texts])
        out = np.empty(len(texts))
        for start in range(0, len(texts), batch_size):
            ids = order[start:start + batch_size]
            batch = service.tokenizer([texts[i] for i in ids], return_tensors="pt", truncation=True,
                                      max_length=MAX_LENGTH, padding=True)
            batch = {k: v.to(service.device) for k, v in batch.items()}
            with torch.inference_mode():
                logits = service.model(**batch).logits
            out[ids] = (logits[:, 1] - logits[:, 0]).cpu().numpy()
        return out

    return score


def main(argv=None, score=None, val: pd.DataFrame | None = None, probabilities=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-legit", type=int, default=MAX_LEGIT, help=f"legitimate postings to sample (1-{MAX_LEGIT})")
    ap.add_argument("--n-boot", type=int, default=N_BOOT, help="group bootstrap resamples")
    args = ap.parse_args(argv)
    if not 1 <= args.n_legit <= MAX_LEGIT or args.n_boot < 1:
        print(f"--n-legit must be between 1 and {MAX_LEGIT}, and --n-boot at least 1.", file=sys.stderr)
        return 2
    val = pd.read_csv(VAL_CSV) if val is None else val
    probabilities = np.load(VAL_PROBABILITIES) if probabilities is None else probabilities
    try:
        result = evaluate(val, probabilities, score or make_real_scorer(), args.n_legit, args.n_boot)
    except (RuntimeError, ValueError) as exc:
        print(f"Stopped: {exc}.", file=sys.stderr)
        return 3
    print("\n".join(report_lines(result)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
