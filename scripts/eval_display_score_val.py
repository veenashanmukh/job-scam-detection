"""Describe display_risk_score on VALIDATION only, from saved probabilities. Writes nothing.

display_risk_score is a monotone display transform of the log-odds, so it cannot change ranking
metrics; this report shows how the 1-99 scale spreads across labels and verdicts.
Do not point this at the test split.

Usage: python scripts/eval_display_score_val.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services.distilbert_service import DEFAULT_THRESHOLD as T  # noqa: E402
from backend.services.job_analysis import display_risk_score, job_verdict  # noqa: E402

MD = ROOT / "backend/models_v2/distilbert_job_scam"
p = np.load(MD / "val_probabilities.npy")
y = np.load(MD / "val_labels.npy")
v = pd.read_csv(ROOT / "data/processed_v2/val.csv")
assert len(v) == len(p) and (v.fraudulent.values == y).all()

# Saved files hold probabilities, not logits; for two classes logit(p) == logit1 - logit0.
pc = np.clip(p, 1e-12, 1 - 1e-12)
logodds = np.log(pc / (1 - pc))
score = np.array([display_risk_score(lo, T) for lo in logodds])
verdict = np.array([job_verdict(pi, T, pr)[0] for pi, pr in zip(p, v.company_profile.fillna(""))])

print(f"threshold={T}  n={len(v)}  frauds={int(y.sum())}")
print(f"score >= 50 agrees with p >= threshold on {np.mean((score >= 50) == (p >= T)):.4f} of rows")
print(f"AUC on probability = {roc_auc_score(y, p):.4f}; AUC on display score = {roc_auc_score(y, score):.4f} "
      "(any difference comes only from ties created by integer truncation)")


def describe(name, s):
    q = np.percentile(s, [0, 10, 50, 90, 100]).astype(int) if len(s) else []
    print(f"{name:<22}{len(s):>6}  min/p10/median/p90/max = {'/'.join(map(str, q))}")


print("\nBy label:")
describe("legitimate", score[y == 0])
describe("fraud", score[y == 1])
print("\nBy verdict:")
for name in ("Likely Legitimate", "Needs Verification", "High Fraud Risk"):
    describe(name, score[verdict == name])

print("\nScore bands (fraud / total, fraud rate):")
edges = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
for lo, hi in zip(edges, edges[1:]):
    m = (score >= lo) & (score < hi)
    rate = f"{y[m].mean():.3f}" if m.any() else "  -  "
    print(f"  {lo:>2}-{hi - 1:<2}  {int(y[m].sum()):>4} / {int(m.sum()):>4}   {rate}")
