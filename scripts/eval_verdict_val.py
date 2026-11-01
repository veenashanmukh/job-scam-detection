"""Evaluate the three-outcome verdict on VALIDATION only, from saved probabilities. Writes nothing.

Exploratory: the verdict rule was chosen after looking at validation errors, so these numbers are
optimistic and are not an independent estimate. Do not point this at the test split.

Usage: python scripts/eval_verdict_val.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services.distilbert_service import DEFAULT_THRESHOLD as T  # noqa: E402
from backend.services.distilbert_service import DistilBERTJobScamService  # noqa: E402
from backend.services.job_analysis import job_verdict  # noqa: E402

MD = ROOT / "backend/models_v2/distilbert_job_scam"
p = np.load(MD / "val_probabilities.npy")
y = np.load(MD / "val_labels.npy")
v = pd.read_csv(ROOT / "data/processed_v2/val.csv")
assert len(v) == len(p) and (v.fraudulent.values == y).all()

# Spot-check alignment: re-score a few rows with the real model.
svc = DistilBERTJobScamService()
idx = list(np.where(y == 1)[0][:4]) + list(np.where(y == 0)[0][:4])
diffs = [abs(svc.predict(v.text[i])["fraud_probability"] - p[i]) for i in idx]
print(f"alignment spot-check on {len(idx)} rows: max |p_saved - p_rescored| = {max(diffs):.2e}")

profile = v.company_profile.fillna("")  # field always present in val.csv; NaN -> empty
verdict = np.array([job_verdict(pi, T, pr)[0] for pi, pr in zip(p, profile)])

print(f"\nthreshold={T}  n={len(v)}  frauds={y.sum()}")
print(f"{'verdict':<20}{'n':>6}{'fraud(TP)':>11}{'legit(FP)':>11}{'fraud rate':>12}")
for name in ("Likely Legitimate", "Needs Verification", "High Fraud Risk"):
    m = verdict == name
    tp = int(y[m].sum())
    fp = int(m.sum() - tp)
    print(f"{name:<20}{m.sum():>6}{tp:>11}{fp:>11}{tp / max(m.sum(), 1):>12.3f}")
flag = p >= T
print(f"{'(all flagged p>=t)':<20}{flag.sum():>6}{int(y[flag].sum()):>11}"
      f"{int(flag.sum() - y[flag].sum()):>11}{y[flag].mean():>12.3f}")
print("note: for Likely Legitimate, 'fraud(TP)' = missed frauds (FN), 'legit(FP)' = true negatives")


def boot(mask, groups=None, B=10000, seed=0):
    """95% percentile bootstrap CI of the fraud rate among rows where mask is True."""
    rng = np.random.default_rng(seed)
    out = []
    if groups is None:
        n = len(v)
        for _ in range(B):
            s = rng.integers(0, n, n)
            m = mask[s]
            if m.sum():
                out.append(y[s][m].mean())
    else:
        ug = np.unique(groups)
        rows = {g: np.where(groups == g)[0] for g in ug}
        for _ in range(B):
            s = np.concatenate([rows[g] for g in rng.choice(ug, len(ug))])
            m = mask[s]
            if m.sum():
                out.append(y[s][m].mean())
    return np.percentile(out, [2.5, 97.5])


hi = verdict == "High Fraud Risk"
k, n = int(y[hi].sum()), int(hi.sum())
g = v.group_id.values
ci = boot(hi)
cg = boot(hi, groups=g)
print(f"\nHigh Fraud Risk precision = {k}/{n} = {k / n:.3f}")
print(f"  95% percentile bootstrap CI (rows, B=10000, seed=0): [{ci[0]:.3f}, {ci[1]:.3f}]")
print(f"  95% cluster bootstrap CI (by group_id):              [{cg[0]:.3f}, {cg[1]:.3f}]")

# The percentile bootstrap collapses to [1, 1] when every row in the tier is fraud,
# so also report the exact Clopper-Pearson interval, on rows and on unique groups.
# Group level: a group counts as correct only if all its High-tier rows are fraud.
group_all_fraud = pd.Series(y[hi]).groupby(g[hi]).min()
for label, N, K in (("rows", n, k), ("unique group_ids", len(group_all_fraud), int(group_all_fraud.sum()))):
    lo = beta.ppf(0.025, K, N - K + 1) if K > 0 else 0.0
    up = beta.ppf(0.975, K + 1, N - K) if K < N else 1.0
    print(f"  Clopper-Pearson 95% ({label}, {K}/{N}):  [{lo:.3f}, {up:.3f}]")

cf = boot(flag)
print(f"For comparison, precision of all flagged = {y[flag].mean():.3f}, "
      f"row-bootstrap CI [{cf[0]:.3f}, {cf[1]:.3f}]")
