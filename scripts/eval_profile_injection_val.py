"""Does adding a company profile to a posting that has none make the classifier stop flagging it? VALIDATION ONLY.

Measurement only: no rule, threshold, model or tier logic is changed. Offline, no LLM, never opens test.csv.

Fixed in advance (see PROFILES, used unchanged): three injected profiles.
Populations (validation): frauds whose cleaned profile is empty; legitimate postings whose cleaned profile is empty
(the false-alarm side); and, as a control, frauds that already have a profile, whose profile is replaced by (B).

Each posting is scored through the same structured path as final_test_eval (ds.build_text of the five fields) with the
deployed DistilBERT and threshold 0.291163. The "before" scores are the saved validation log-odds (the scorer is
checked against them on the first rows). The verdict uses the app's own path: job_verdict with the cleaned profile, then
escalate_verdict on the rule signs found in the text that was scored.

Per population and profile: n, flagged before/after, mean log-odds shift with a 95% group bootstrap interval (group_id,
2000 resamples, seed 42), and how many end as Needs Verification or High Fraud Risk.

Usage: python scripts/eval_profile_injection_val.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from backend.services import dataset as ds  # noqa: E402
from backend.services.job_analysis import (  # noqa: E402
    VERDICT_LEGITIMATE, detect_warning_signs, escalate_verdict, job_verdict,
)

BANNER = "VALIDATION ONLY - ADVERSARIAL PROFILE INJECTION. Not a test-set result."
THRESHOLD = 0.291163
SEED = 42
N_BOOT = 2000
ALIGNMENT_ROWS = 32
ALIGNMENT_TOLERANCE = 1e-3

# Written before any result was seen and never changed.
PROFILES = {
    "A": "x",
    "B": "We are a growing company with a friendly team. We value honesty and hard work.",
    "C": ("Founded in 2005, Northfield Group is a established provider of business services to clients across several "
          "industries. Our team of experienced professionals is committed to quality, integrity and long-term "
          "relationships. We offer a supportive work environment, ongoing training and opportunities for career growth. "
          "Northfield Group is an equal opportunity employer."),
}
VAL_CSV = ROOT / "data" / "processed_v2" / "val.csv"
VAL_PROBABILITIES = ROOT / "backend" / "models_v2" / "distilbert_job_scam" / "val_probabilities.npy"


def cleaned_profile(row: dict) -> str:
    return ds.clean_field(row.get("company_profile")) or ""


def text_with_profile(row: dict, profile) -> str:
    """ds.build_text with the row's fields and the given company profile (None or "" adds no profile line)."""
    fields = {field: row.get(field) for field in ds.TEXT_FIELDS}
    fields["company_profile"] = profile
    return ds.build_text(pd.Series(fields))


def app_verdict(probability: float, text: str, profile: str) -> str:
    """The verdict the app would give: classifier plus profile check, then escalation on the rules found in text."""
    verdict, reason = job_verdict(float(probability), THRESHOLD, profile)
    return escalate_verdict(verdict, reason, detect_warning_signs(text))[0]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def bootstrap_mean(values: np.ndarray, groups: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float]:
    """95% interval of the mean over group_id resamples."""
    if len(values) == 0:
        return float("nan"), float("nan")
    index = list(pd.Series(np.arange(len(groups))).groupby(np.asarray(groups)).indices.values())
    rng = np.random.default_rng(SEED)
    means = []
    for _ in range(n_boot):
        picked = np.concatenate([index[i] for i in rng.integers(0, len(index), len(index))])
        means.append(values[picked].mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def summarise(rows, before_logodds, after_logodds, after_texts, after_profiles, groups, n_boot) -> dict:
    before_p, after_p = sigmoid(before_logodds), sigmoid(after_logodds)
    shift = after_logodds - before_logodds
    caught_before = sum(app_verdict(p, text_with_profile(r, r.get("company_profile")), cleaned_profile(r)) != VERDICT_LEGITIMATE
                        for r, p in zip(rows, before_p))
    caught_after = sum(app_verdict(p, t, prof) != VERDICT_LEGITIMATE for p, t, prof in zip(after_p, after_texts, after_profiles))
    lo, hi = bootstrap_mean(shift, groups, n_boot)
    return {"n": len(rows), "flagged_before": int((before_p >= THRESHOLD).sum()), "flagged_after": int((after_p >= THRESHOLD).sum()),
            "mean_shift": float(shift.mean()) if len(shift) else float("nan"), "shift_ci95": [lo, hi],
            "verdict_caught_before": int(caught_before), "verdict_caught_after": int(caught_after)}


def evaluate(val: pd.DataFrame, probabilities, score, n_boot: int = N_BOOT) -> dict:
    """score(texts) -> log-odds array from the deployed model (or a fake in tests)."""
    if len(val) != len(probabilities):
        raise ValueError("val.csv and val_probabilities.npy have different lengths")
    rows = val.to_dict("records")
    baseline = logit(probabilities)

    probe = list(range(min(ALIGNMENT_ROWS, len(rows))))
    gap = float(np.max(np.abs(np.asarray(score([text_with_profile(rows[i], rows[i].get("company_profile")) for i in probe])) - baseline[probe])))
    if gap > ALIGNMENT_TOLERANCE:
        raise RuntimeError(f"the scorer does not reproduce the saved validation log-odds (max gap {gap:.2e})")

    y = val[ds.LABEL_COL].to_numpy(dtype=int)
    empty = np.array([cleaned_profile(r) == "" for r in rows])
    groups = val[ds.GROUP_COL].to_numpy()
    populations = {"frauds with no profile": np.flatnonzero((y == 1) & empty),
                   "legitimate with no profile": np.flatnonzero((y == 0) & empty)}
    result = {"alignment_gap": gap, "populations": {}, "control": {}}
    for name, idx in populations.items():
        result["populations"][name] = {}
        for key, profile in PROFILES.items():
            texts = [text_with_profile(rows[i], profile) for i in idx]
            after = np.asarray(score(texts), dtype=float)
            result["populations"][name][key] = summarise([rows[i] for i in idx], baseline[idx], after, texts,
                                                         [ds.clean_field(profile) or ""] * len(idx), groups[idx], n_boot)
    control = np.flatnonzero((y == 1) & ~empty)
    texts = [text_with_profile(rows[i], PROFILES["B"]) for i in control]
    after = np.asarray(score(texts), dtype=float)
    result["control"]["B"] = summarise([rows[i] for i in control], baseline[control], after, texts,
                                       [PROFILES["B"]] * len(control), groups[control], n_boot)
    return result


def _line(label: str, s: dict, caught_word: str) -> str:
    lo, hi = s["shift_ci95"]
    return (f"  {label:<4}{s['n']:>5}{s['flagged_before']:>9}{s['flagged_after']:>8}   {s['mean_shift']:>+8.3f}  "
            f"[{lo:+.3f}, {hi:+.3f}]   {s['verdict_caught_before']:>4} -> {s['verdict_caught_after']:<4} {caught_word}")


def report_lines(result: dict) -> list[str]:
    words = {"frauds with no profile": "caught", "legitimate with no profile": "false alarms"}
    lines = [BANNER, f"threshold {THRESHOLD}; before-scores are the saved validation log-odds (scorer max gap {result['alignment_gap']:.1e})",
             "Injected profiles (fixed in advance):"]
    for key, text in PROFILES.items():
        count = len(text.split())
        lines.append(f"  ({key}) {count} word{'s' if count != 1 else ''}: {text!r}")
    header = f"  {'':<4}{'n':>5}{'flagged':>9}{'flagged':>8}   {'mean log-odds':>13}  {'95% group bootstrap':<20}  verdict Needs Verification or High Fraud Risk"
    sub = f"  {'':<4}{'':>5}{'before':>9}{'after':>8}   {'shift':>13}"
    for name, by_profile in result["populations"].items():
        lines += ["", f"{name}:", header, sub]
        lines += [_line(f"({k})", s, words[name]) for k, s in by_profile.items()]
    lines += ["", "Control: frauds that already have a profile, profile replaced by (B):", header, sub]
    lines += [_line("(B)", s, "caught") for s in result["control"].values()]
    return lines + ["", BANNER]


def main() -> int:
    from eval_input_formats_val import make_real_scorer

    val = pd.read_csv(VAL_CSV)
    try:
        result = evaluate(val, np.load(VAL_PROBABILITIES), make_real_scorer())
    except (RuntimeError, ValueError) as exc:
        print(f"Stopped: {exc}.", file=sys.stderr)
        return 3
    print("\n".join(report_lines(result)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
