"""Measurement only: how would the CHAT rule engine score the validation job postings? VALIDATION ONLY, offline.

Each validation posting is turned into plain text (its cleaned fields joined with newlines, no labels, as pasted) and
scored by the chat engine (backend/services/conversation_analysis.analyze_messages) as ONE message from one other
person. Reported at each chat risk level (Higher, Moderate, Lower): how many legitimate postings it would flag and how
many frauds it would catch, for all postings and for the postings the job analyzer currently calls "Likely Legitimate"
(the population a dual run could change). Nothing is wired into the app and no rule is changed.

Pre-registered bar (same as for the single-category escalation): a dual run would be adopted only if it raises at most
10 false alarms among the legitimate postings currently "Likely Legitimate" (2,537 on validation).

Usage: python scripts/eval_chat_rules_on_val.py
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services import dataset as ds  # noqa: E402
from backend.services.conversation_analysis import analyze_messages  # noqa: E402
from backend.services.job_analysis import VERDICT_LEGITIMATE, detect_warning_signs, escalate_verdict, job_verdict  # noqa: E402

BANNER = "VALIDATION ONLY - MEASUREMENT OF THE CHAT RULES ON JOB POSTINGS. Nothing is wired into the app."
THRESHOLD = 0.291163
MAX_FALSE_ALARMS = 10
LEVELS = ("Higher risk", "Moderate risk", "Lower risk")
VAL_CSV = ROOT / "data" / "processed_v2" / "val.csv"
VAL_PROBABILITIES = ROOT / "backend" / "models_v2" / "distilbert_job_scam" / "val_probabilities.npy"


def plain_text(row: dict) -> str:
    cleaned = (ds.clean_field(row.get(field)) for field in ds.TEXT_FIELDS)
    return "\n".join(value for value in cleaned if value)


def chat_result(text: str) -> dict:
    """The chat engine's report for the text as a single message from one other person."""
    return analyze_messages([{"time": "", "speaker": "Other", "text": text, "is_system": False}])


def compute(val: pd.DataFrame, probabilities, threshold: float = THRESHOLD) -> dict:
    if len(val) != len(probabilities):
        raise ValueError("val.csv and val_probabilities.npy have different lengths")
    y = val[ds.LABEL_COL].to_numpy(dtype=int)
    levels, signs, likely = [], [], []
    for row, p in zip(val.to_dict("records"), probabilities):
        text = plain_text(row)
        report = chat_result(text)
        levels.append(report["risk_category"])
        signs.append(report["warning_signs"])
        verdict, reason = job_verdict(float(p), threshold, ds.clean_field(row.get("company_profile")) or "")
        likely.append(escalate_verdict(verdict, reason, detect_warning_signs(text))[0] == VERDICT_LEGITIMATE)
    levels, likely = np.array(levels), np.array(likely)
    populations = {
        "legitimate, all": y == 0,
        "legitimate, currently Likely Legitimate": (y == 0) & likely,
        "fraud, all": y == 1,
        "fraud, currently Likely Legitimate (missed)": (y == 1) & likely,
    }
    table = {name: {level: int(((levels == level) & mask).sum()) for level in LEVELS} | {"n": int(mask.sum())}
             for name, mask in populations.items()}
    flagged_legit = (levels != LEVELS[2]) & populations["legitimate, currently Likely Legitimate"]
    causes = Counter(sign for s, flag in zip(signs, flagged_legit) if flag for sign in s)
    return {"n": len(val), "table": table, "causes_of_false_alarms": causes}


def cumulative(row: dict) -> dict:
    return {"Higher only": row["Higher risk"], "Moderate or higher": row["Higher risk"] + row["Moderate risk"]}


def report_lines(result: dict) -> list[str]:
    t = result["table"]
    lines = [BANNER, f"{result['n']} validation postings scored as one plain-text chat message each", "",
             f"{'population':<46}{'n':>6}{'Higher':>8}{'Moderate':>10}{'Lower':>8}"]
    for name, row in t.items():
        lines.append(f"{name:<46}{row['n']:>6}{row['Higher risk']:>8}{row['Moderate risk']:>10}{row['Lower risk']:>8}")
    legit_ll, fraud_ll = t["legitimate, currently Likely Legitimate"], t["fraud, currently Likely Legitimate (missed)"]
    lines += ["", "Cumulative flags (chat level reached):"]
    for cutoff in ("Higher only", "Moderate or higher"):
        la, lb = cumulative(t["legitimate, all"])[cutoff], cumulative(legit_ll)[cutoff]
        fa, fb = cumulative(t["fraud, all"])[cutoff], cumulative(fraud_ll)[cutoff]
        verdict = "within" if lb <= MAX_FALSE_ALARMS else "over"
        lines.append(f"  {cutoff:<20} legitimate flagged {la}/{t['legitimate, all']['n']} (of the {legit_ll['n']} currently "
                     f"Likely Legitimate: {lb}); frauds caught {fa}/{t['fraud, all']['n']} (of the {fraud_ll['n']} currently "
                     f"missed: {fb})  -> false alarms {lb} vs bar <= {MAX_FALSE_ALARMS} of {legit_ll['n']}: {verdict} the bar")
    if result["causes_of_false_alarms"]:
        lines += ["", "Rules behind the false alarms among the currently Likely Legitimate (Moderate or higher): "
                  + ", ".join(f"{k} {v}" for k, v in result["causes_of_false_alarms"].most_common())]
    return lines + ["", BANNER]


def main() -> int:
    from backend.services.distilbert_service import DEFAULT_THRESHOLD

    val = pd.read_csv(VAL_CSV)
    print("\n".join(report_lines(compute(val, np.load(VAL_PROBABILITIES), DEFAULT_THRESHOLD))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
