"""What the escalation-only verdict rule would change on VALIDATION postings. Offline, writes nothing.

Uses val_probabilities.npy and the warning-sign rules on val.csv (never test.csv). For every validation posting:
the classifier-only verdict (job_verdict) versus the verdict with escalation (escalate_verdict: "Likely
Legitimate" becomes "Needs Verification" when at least 2 distinct rule categories fire, or when one of the three
specific categories fires alone). Reports how many postings
change, split by true label (the false-alarm cost versus the frauds recovered), and the verdict-tier table before
and after. The rules were extended after looking at a synthetic development set, not at validation, so this is a
validation check, but the escalation threshold was not chosen on it.

Usage: python scripts/eval_escalation_val.py [--single-category]
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services import dataset as ds  # noqa: E402
from backend.services.job_analysis import (  # noqa: E402
    VERDICT_LEGITIMATE, detect_warning_signs, escalate_verdict, job_verdict,
)

BANNER = "VALIDATION ONLY - ESCALATION RULE COST/BENEFIT. Not a test-set result."
VAL_CSV = ROOT / "data" / "processed_v2" / "val.csv"
PROBABILITIES = ROOT / "backend" / "models_v2" / "distilbert_job_scam" / "val_probabilities.npy"
TIERS = ("Likely Legitimate", "Needs Verification", "High Fraud Risk")
SPECIFIC_LABELS = ("Cheque or overpayment scam", "Gift card or crypto payment", "Parcel or money forwarding")
SINGLE_CATEGORY_BAR = 10  # pre-registered: adopt single-category escalation only if legitimate false alarms are <= 10


def compute(val: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> dict:
    if len(val) != len(probabilities):
        raise ValueError("val.csv and val_probabilities.npy have different lengths")
    y = val[ds.LABEL_COL].to_numpy(dtype=int)
    before, after, categories = [], [], []
    for row, p in zip(val.to_dict("records"), probabilities):
        signs = detect_warning_signs("" if pd.isna(row.get("text")) else str(row["text"]))
        verdict, reason = job_verdict(float(p), threshold, ds.clean_field(row.get("company_profile")) or "")
        before.append(verdict)
        after.append(escalate_verdict(verdict, reason, signs)[0])
        categories.append([s["label"] for s in signs])
    before, after = np.array(before), np.array(after)
    changed = before != after
    table = []
    for tier in TIERS:
        row = {"tier": tier}
        for name, verdicts in (("before", before), ("after", after)):
            mask = verdicts == tier
            row[name] = {"n": int(mask.sum()), "fraud": int(y[mask].sum())}
        table.append(row)
    in_tier = before == VERDICT_LEGITIMATE
    # Measurement only: postings now "Likely Legitimate" where one specific category fires and no other does.
    distinct = [set(c) for c in categories]
    single = {}
    for label in SPECIFIC_LABELS:
        mask = in_tier & np.array([d == {label} for d in distinct])
        single[label] = {"legit": int((y[mask] == 0).sum()), "fraud": int((y[mask] == 1).sum())}
    any_mask = in_tier & np.array([len(d) == 1 and next(iter(d)) in SPECIFIC_LABELS for d in distinct])
    single_category = {"labels": single, "legit_total": int((y[any_mask] == 0).sum()),
                       "fraud_total": int((y[any_mask] == 1).sum()), "bar_max_false_alarms": SINGLE_CATEGORY_BAR}
    return {
        "single_category": single_category,
        "n": len(val), "n_fraud": int(y.sum()), "threshold": threshold,
        "changed": int(changed.sum()), "changed_fraud": int(y[changed].sum()),
        "changed_legit": int((y[changed] == 0).sum()),
        "likely_legitimate_before": int(in_tier.sum()), "legit_in_tier": int((y[in_tier] == 0).sum()),
        "fraud_in_tier": int(y[in_tier].sum()), "tiers": table,
        "category_combinations": Counter(" + ".join(c) for c, flag in zip(categories, changed) if flag),
        "postings_with_any_sign": int(sum(1 for c in categories if c)),
        "postings_with_two_or_more": int(sum(1 for c in categories if len(set(c)) >= 2)),
    }


def report_lines(r: dict) -> list[str]:
    lines = [BANNER, f"{r['n']} validation postings ({r['n_fraud']} fraud), threshold {r['threshold']}", "",
             f"Rules fire on >= 1 category for {r['postings_with_any_sign']} postings and on >= 2 for "
             f"{r['postings_with_two_or_more']}.",
             f"Postings the escalation rule would change: {r['changed']} "
             f"(legitimate = false alarms: {r['changed_legit']}; fraud = recovered: {r['changed_fraud']})"]
    if r["likely_legitimate_before"]:
        lines.append(f"Of the {r['likely_legitimate_before']} 'Likely Legitimate' postings, "
                     f"{r['legit_in_tier']} are legitimate and {r['fraud_in_tier']} are missed frauds: "
                     f"{r['changed_legit']} / {r['legit_in_tier']} legitimate postings would be escalated "
                     f"({r['changed_legit'] / max(r['legit_in_tier'], 1):.1%}); "
                     f"{r['changed_fraud']} / {r['fraud_in_tier']} missed frauds recovered "
                     f"({r['changed_fraud'] / max(r['fraud_in_tier'], 1):.1%}).")
    lines += ["", f"{'verdict tier':<22}{'before n':>9}{'fraud':>7}{'rate':>8}   {'after n':>9}{'fraud':>7}{'rate':>8}"]
    for t in r["tiers"]:
        b, a = t["before"], t["after"]
        rate = lambda x: f"{x['fraud'] / x['n']:.3f}" if x["n"] else "   -  "  # noqa: E731
        lines.append(f"{t['tier']:<22}{b['n']:>9}{b['fraud']:>7}{rate(b):>8}   {a['n']:>9}{a['fraud']:>7}{rate(a):>8}")
    if r["category_combinations"]:
        lines += ["", "Category combinations behind the changes:"]
        lines += [f"  {n:>3} x {combo}" for combo, n in r["category_combinations"].most_common()]
    lines += ["", BANNER]
    return lines


def single_category_lines(r: dict) -> list[str]:
    s = r["single_category"]
    lines = ["", "Single-category measurement (no rule or verdict is changed): postings now 'Likely Legitimate' where "
                 "the label fires and no other category does",
             f"{'label':<30}{'legitimate (false alarms)':>27}{'fraud (recovered)':>20}"]
    for label, counts in s["labels"].items():
        lines.append(f"{label:<30}{counts['legit']:>27}{counts['fraud']:>20}")
    lines += [f"{'all three':<30}{s['legit_total']:>27}{s['fraud_total']:>20}",
              f"Of {r['legit_in_tier']} legitimate and {r['fraud_in_tier']} fraud postings currently 'Likely Legitimate'.",
              f"Pre-registered bar: adopt only if legitimate false alarms <= {s['bar_max_false_alarms']} of "
              f"{r['legit_in_tier']}; observed {s['legit_total']}."]
    return lines


def main(argv=None) -> int:
    import argparse

    from backend.services.distilbert_service import DEFAULT_THRESHOLD

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--single-category", action="store_true",
                    help="also report how many 'Likely Legitimate' postings fire one specific category alone")
    args = ap.parse_args(argv)
    val = pd.read_csv(VAL_CSV)
    probabilities = np.load(PROBABILITIES)
    result = compute(val, probabilities, DEFAULT_THRESHOLD)
    lines = report_lines(result)
    if args.single_category:
        lines = lines[:-1] + single_category_lines(result) + ["", BANNER]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
