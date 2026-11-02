"""Per-rule precision and recall of the job-posting warning-sign rules on VALIDATION only. Writes nothing.

Measures the rules as they are (backend/services/job_analysis.py::WARNING_RULES) on val.csv
`text`. The candidate-phrase table is a diagnostic for planning new rules, not a rule set.
EMSCAD postings date from 2012-2014, so modern scam phrasing (Telegram, crypto) may be rare;
low counts here do not mean a sign is unimportant in today's postings.

Usage: python scripts/eval_rules_val.py
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services.job_analysis import WARNING_RULES, detect_warning_signs  # noqa: E402

# Diagnostic only: how often each phrase family appears by label.
CANDIDATE_PHRASES = {
    "deposit": r"\bdeposits?\b",
    "refundable": r"\brefundable\b",
    "registration/training/starter fee": r"\b(?:registration|training|starter|processing)\s+(?:fee|charge|kit)s?\b",
    "telegram/whatsapp/signal": r"\b(?:telegram|whatsapp|signal app|wechat)\b",
    "text/sms us": r"\b(?:text|sms)\s+(?:us|me)\b",
    "urgent": r"\burgent(?:ly)?\b",
}


def main():
    v = pd.read_csv(ROOT / "data" / "processed_v2" / "val.csv")
    y = v["fraudulent"].to_numpy()
    n_fraud, n_legit = int(y.sum()), int((y == 0).sum())
    fired = pd.DataFrame(
        [{sign["label"]: True for sign in detect_warning_signs(text)} for text in v["text"].fillna("")],
        columns=[label for label, _ in WARNING_RULES],
    ).fillna(False).astype(bool)

    print("=" * 76)
    print(f"VALIDATION ONLY - rule evaluation on {len(v)} postings ({n_fraud} fraud, {n_legit} legitimate)")
    print("=" * 76)
    print(f"{'rule':<32}{'fires':>6}{'fraud':>7}{'legit':>7}{'precision':>11}{'recall':>8}{'legit FPR':>11}")
    rows = [(label, fired[label].to_numpy()) for label, _ in WARNING_RULES]
    rows.append(("ANY RULE", fired.any(axis=1).to_numpy()))
    for label, mask in rows:
        tp, fp = int((mask & (y == 1)).sum()), int((mask & (y == 0)).sum())
        precision = f"{tp / (tp + fp):.3f}" if tp + fp else "  -  "
        print(f"{label:<32}{int(mask.sum()):>6}{tp:>7}{fp:>7}{precision:>11}{tp / n_fraud:>8.3f}{fp / n_legit:>11.4f}")
    print(f"\nbase rate (precision of guessing fraud): {n_fraud / len(v):.3f}")

    print("\nCandidate phrase frequency (diagnostic, NOT rules):")
    print(f"{'phrase family':<36}{'fraud':>7}{'legit':>7}{'share fraud':>13}")
    text = v["text"].fillna("")
    for name, pattern in CANDIDATE_PHRASES.items():
        mask = text.str.contains(pattern, flags=re.IGNORECASE, regex=True).to_numpy()
        tp, fp = int((mask & (y == 1)).sum()), int((mask & (y == 0)).sum())
        share = f"{tp / (tp + fp):.3f}" if tp + fp else "  -  "
        print(f"{name:<36}{tp:>7}{fp:>7}{share:>13}")
    print("\nVALIDATION ONLY. Counts are small; treat precision on fewer than ~10 firings as unreliable.")


if __name__ == "__main__":
    main()
