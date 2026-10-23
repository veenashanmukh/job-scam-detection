"""Train the TF-IDF + Logistic Regression baseline on train, tune on val. NEVER touches the test split.

Usage (repo root, venv active):
    python scripts/verify_splits.py --dir data/processed        # first: confirm splits are fresh and leakage-free
    python scripts/train_baseline.py --person YOURNAME --log-results

Threshold rule (decide BEFORE looking at results; default max-F1):
    --min-precision 0.80   -> instead choose max recall subject to validation precision >= 0.80
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services import baseline as bl  # noqa: E402
from backend.services import dataset as ds  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", default=str(ROOT / "data/processed"))
    ap.add_argument("--out", default=str(ROOT / "reports/baseline"))
    ap.add_argument("--model-dir", default=str(ROOT / "backend/models"))
    ap.add_argument("--c-grid", default=",".join(str(c) for c in bl.DEFAULT_C_GRID))
    ap.add_argument("--min-precision", type=float, default=None)
    ap.add_argument("--seed", type=int, default=ds.SEED)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--person", default=None, help="name to record in results.csv")
    ap.add_argument("--log-results", action="store_true", help="append validation metrics to data/evaluation/results.csv")
    args = ap.parse_args()

    rule = "recall_at_precision" if args.min_precision is not None else "max_f1"
    try:
        rec = bl.run_baseline(args.processed, args.out, args.model_dir,
                              c_grid=tuple(float(c) for c in args.c_grid.split(",")), rule=rule,
                              min_precision=args.min_precision, seed=args.seed, n_boot=args.n_bootstrap)
    except (RuntimeError, ValueError, PermissionError, FileNotFoundError) as e:
        print(f"TRAINING REFUSED/FAILED: {e}", file=sys.stderr)
        return 4

    if rec["pipeline_code_sha256_in_manifest"] != ds.file_sha256(ds.__file__):
        print("WARNING: dataset.py has changed since these splits were made. "
              "Run scripts/verify_splits.py and regenerate if it fails.", file=sys.stderr)

    v, ci = rec["val"], rec["val_bootstrap_95ci"]
    print(f"splits: train={rec['split_sizes']['train']} val={rec['split_sizes']['val']} | vocab={rec['vocab_size']}")
    for row in rec["c_grid_results"]:
        mark = "  <- chosen" if row["C"] == rec["chosen_C"] else ""
        print(f"  C={row['C']:<5} val PR-AUC={row['val_average_precision']:.4f}{mark}")
    print(f"threshold ({rec['threshold_rule']}) = {rec['frozen_threshold']:.4f}  [chosen on validation, now frozen]")
    print("VALIDATION, fraud class (label 1):")
    for k in ("precision", "recall", "f1", "average_precision"):
        print(f"  {k:<18} {v[k]:.4f}   95% CI [{ci[k][0]:.3f}, {ci[k][1]:.3f}]")
    print(f"  ROC-AUC            {v['roc_auc']:.4f}")
    print(f"  confusion [[TN FP],[FN TP]] = [[{v['tn']}, {v['fp']}], [{v['fn']}, {v['tp']}]]  (fraud n={v['n_fraud']})")
    t = rec["train_at_same_threshold"]
    print(f"  train (same threshold): precision={t['precision']:.3f} recall={t['recall']:.3f}  <- big gap vs val = overfitting")
    print(f"saved: {Path(args.out) / 'baseline_run.json'}  and  {Path(args.model_dir) / bl.MODEL_FILE}")

    if args.log_results:
        if not args.person:
            print("--log-results needs --person", file=sys.stderr)
            return 2
        bl.append_results(ROOT / "data/evaluation/results.csv", args.person, "baseline_val", "tfidf_lr",
                          {"val_fraud_precision": v["precision"], "val_fraud_recall": v["recall"],
                           "val_fraud_f1": v["f1"], "val_pr_auc": v["average_precision"], "val_roc_auc": v["roc_auc"]})
        print("appended validation metrics to data/evaluation/results.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
