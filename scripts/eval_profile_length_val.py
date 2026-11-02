"""Measurement only: company-profile length among the validation postings the classifier flags. VALIDATION ONLY.

Uses val.csv and the saved validation probabilities (no model is run). Among postings flagged at the deployed
threshold, reports how many have a cleaned company profile of 0, 1-19, 20-99 and 100+ characters (clean_field: tags
removed, entities decoded, spaces collapsed; a missing profile is 0), with the fraud count in each bucket. The tier
rule (a flagged posting with any non-empty profile is "High Fraud Risk") is not changed or evaluated here.

Usage: python scripts/eval_profile_length_val.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services import dataset as ds  # noqa: E402

BANNER = "VALIDATION ONLY - PROFILE LENGTH AMONG FLAGGED POSTINGS. Measurement only; no tier rule is changed."
THRESHOLD = 0.291163
BUCKETS = (("0", 0, 0), ("1-19", 1, 19), ("20-99", 20, 99), ("100+", 100, None))
VAL_CSV = ROOT / "data" / "processed_v2" / "val.csv"
VAL_PROBABILITIES = ROOT / "backend" / "models_v2" / "distilbert_job_scam" / "val_probabilities.npy"


def cleaned_length(profile) -> int:
    return len(ds.clean_field(profile) or "")


def bucket_of(length: int) -> str:
    for name, low, high in BUCKETS:
        if length >= low and (high is None or length <= high):
            return name
    raise ValueError(length)


def compute(val: pd.DataFrame, probabilities, threshold: float = THRESHOLD) -> dict:
    if len(val) != len(probabilities):
        raise ValueError("val.csv and val_probabilities.npy have different lengths")
    flagged = np.asarray(probabilities) >= threshold
    y = val[ds.LABEL_COL].to_numpy(dtype=int)
    lengths = np.array([cleaned_length(p) for p in val["company_profile"]])
    names = np.array([bucket_of(int(n)) for n in lengths])
    table = {}
    for name, _, _ in BUCKETS:
        mask = flagged & (names == name)
        n, fraud = int(mask.sum()), int(y[mask].sum())
        table[name] = {"n": n, "fraud": fraud, "legitimate": n - fraud}
    return {"n_val": len(val), "n_flagged": int(flagged.sum()), "n_flagged_fraud": int(y[flagged].sum()), "buckets": table}


def report_lines(result: dict) -> list[str]:
    lines = [BANNER, f"{result['n_val']} validation postings; {result['n_flagged']} flagged at threshold {THRESHOLD} "
                     f"({result['n_flagged_fraud']} fraud)", "",
             f"{'cleaned profile length':<24}{'flagged':>9}{'fraud':>8}{'legitimate':>12}{'fraud share':>13}"]
    for name, row in result["buckets"].items():
        share = f"{row['fraud'] / row['n']:.3f}" if row["n"] else "-"
        lines.append(f"{name + ' characters':<24}{row['n']:>9}{row['fraud']:>8}{row['legitimate']:>12}{share:>13}")
    return lines + ["", BANNER]


def main() -> int:
    val = pd.read_csv(VAL_CSV)
    print("\n".join(report_lines(compute(val, np.load(VAL_PROBABILITIES)))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
