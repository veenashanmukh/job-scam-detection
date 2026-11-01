"""Score the synthetic job postings with the deployed DistilBERT service and the warning-sign rules. Offline: no LLM.

Reads <data-dir>/postings.json (default data/synthetic_jobs). Each posting is turned into the classifier text with
the shared build_text (title, company_profile, description, requirements, benefits), scored by the deployed
DistilBERT service, and checked by the rules. The verdict is reported twice: the classifier-only verdict
(job_verdict) and the verdict with the escalation-only rule (escalate_verdict). A posting counts as predicted scam
when its verdict is not "Likely Legitimate".

Usage: python scripts/eval_synthetic_jobs.py [--data-dir DIR]
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services import dataset as ds  # noqa: E402
from backend.services.job_analysis import (  # noqa: E402
    VERDICT_LEGITIMATE, detect_warning_signs, display_risk_score, escalate_verdict, job_verdict,
)

DEFAULT_DIR = ROOT / "data" / "synthetic_jobs"
BANNER = "SYNTHETIC SET - BEHAVIOUR DEMONSTRATION, NOT REAL-WORLD ACCURACY"
VARIANTS = (("without escalation", "verdict_plain"), ("with escalation", "verdict_escalated"))


def load_postings(directory: Path) -> list[dict]:
    postings = json.loads((directory / "postings.json").read_text(encoding="utf-8"))
    counts = {k: sum(1 for p in postings if p["label"] == k) for k in ("scam", "benign")}
    if not counts["scam"] or not counts["benign"]:
        raise ValueError(f"postings.json needs at least 1 scam and 1 benign posting; found {counts}.")
    return postings


def posting_text(posting: dict) -> str:
    return ds.build_text(pd.Series({field: posting.get(field) for field in ds.TEXT_FIELDS}))


def score_postings(postings: list[dict], predict) -> list[dict]:
    """predict(text) -> {"fraud_probability", "fraud_logodds", "threshold"} (the DistilBERT service's output)."""
    rows = []
    for posting in postings:
        text = posting_text(posting)
        prediction = predict(text)
        probability, threshold = float(prediction["fraud_probability"]), float(prediction["threshold"])
        profile = ds.clean_field(posting.get("company_profile")) or ""
        signs = detect_warning_signs(text)
        plain, plain_reason = job_verdict(probability, threshold, profile)
        escalated, escalated_reason = escalate_verdict(plain, plain_reason, signs)
        rows.append({**{k: posting[k] for k in ("id", "type", "label", "difficulty")},
                     "fraud_probability": probability,
                     "display_score": display_risk_score(prediction.get("fraud_logodds"), threshold),
                     "categories": [s["label"] for s in signs],
                     "verdict_plain": plain, "verdict_escalated": escalated, "escalation_reason": escalated_reason})
    return rows


def predicted_scam(row: dict, key: str) -> bool:
    return row[key] != VERDICT_LEGITIMATE


def metrics(rows: list[dict], key: str) -> dict:
    tp = sum(1 for r in rows if r["label"] == "scam" and predicted_scam(r, key))
    fn = sum(1 for r in rows if r["label"] == "scam" and not predicted_scam(r, key))
    fp = sum(1 for r in rows if r["label"] != "scam" and predicted_scam(r, key))
    tn = len(rows) - tp - fn - fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"accuracy": (tp + tn) / len(rows), "precision": precision, "recall": recall,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def report_lines(rows: list[dict], directory: Path) -> list[str]:
    scam = sum(1 for r in rows if r["label"] == "scam")
    out = ["=" * 70, BANNER, "=" * 70, f"Data directory: {directory}",
           f"{len(rows)} postings ({scam} scam, {len(rows) - scam} benign); DistilBERT + rules, no LLM.", ""]
    for name, key in VARIANTS:
        m = metrics(rows, key)
        out.append(f"{name:<19} accuracy {m['accuracy']:.3f}  precision {m['precision']:.3f}  recall {m['recall']:.3f}"
                   f"  (TP {m['tp']} FP {m['fp']} TN {m['tn']} FN {m['fn']})")
    out += ["", "Rule categories per posting (probability / display score / verdict without -> with escalation):"]
    for r in rows:
        cats = ", ".join(r["categories"]) or "none"
        out.append(f"  {r['id']:<8}{r['label']:<7}{r['type']:<18} p={r['fraud_probability']:.3f} score={r['display_score']}"
                   f"  {r['verdict_plain']} -> {r['verdict_escalated']}  [{cats}]")
    for name, key in VARIANTS:
        wrong = [r for r in rows if predicted_scam(r, key) != (r["label"] == "scam")]
        out += ["", f"Misclassified {name}: {len(wrong)}"]
        for r in wrong:
            out.append(f"  {r['id']} type={r['type']} label={r['label']} difficulty={r['difficulty']} "
                       f"verdict={r[key]} p={r['fraud_probability']:.3f} categories={r['categories'] or 'none'}")
    return out


def main(argv=None, predict=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DIR, help="folder with postings.json")
    args = ap.parse_args(argv)
    try:
        postings = load_postings(args.data_dir)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    if predict is None:
        from backend.services.distilbert_service import DistilBERTJobScamService
        predict = DistilBERTJobScamService().predict
    rows = score_postings(postings, predict)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # the data path may hold characters the console cannot show
    print("\n".join(report_lines(rows, args.data_dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
