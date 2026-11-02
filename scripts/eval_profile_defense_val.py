"""Would a "strip score" defence stop the company profile from hiding scams? VALIDATION ONLY. Measurement only.

No rule, threshold, model, tier logic, app code or final_test_eval.py is changed; offline, no LLM, never opens test.csv.

Defence under test: for each posting take the log-odds with the posting's own profile (the saved validation log-odds)
and the log-odds with the company profile removed (same build_text path, everything else unchanged). The defended
score is the larger of the two and is compared with the same threshold, in log-odds. A posting with an empty profile is
unchanged by definition. After an injected profile the "profile removed" text is the original empty-profile text, so
its score is the saved one.

Pre-registered adoption bar (fixed before any result was seen):
  (1) legitimate false alarms on full validation rise by at most 10 over the deployed count;
  (2) after each injection A, B, C (the fixed profiles of eval_profile_injection_val.py), at least 43 of the 53
      originally flagged frauds with an empty profile are still flagged under the defended score;
  (3) the defended validation PR-AUC is not more than 0.02 below the deployed PR-AUC.

Tiers under the defence: High Fraud Risk only if the deployed (with-profile) score flags the posting and a profile is
present; flagged only through the strip score is Needs Verification.

Scores are cached on disk (reports/profile_runs/, gitignored) so an interrupted run resumes, and progress goes to stderr.

Usage: python scripts/eval_profile_defense_val.py
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from backend.services import dataset as ds  # noqa: E402
from backend.services.job_analysis import (  # noqa: E402
    VERDICT_HIGH_RISK, VERDICT_LEGITIMATE, VERDICT_NEEDS_VERIFICATION, detect_warning_signs, escalate_verdict,
    job_verdict,
)
from eval_profile_injection_val import (  # noqa: E402
    ALIGNMENT_ROWS, ALIGNMENT_TOLERANCE, N_BOOT, PROFILES, SEED, THRESHOLD, bootstrap_mean, cleaned_profile, logit,
    text_with_profile,
)

BANNER = "VALIDATION ONLY - PROFILE DEFENCE. Not a test-set result."
THRESHOLD_LOGODDS = float(np.log(THRESHOLD / (1 - THRESHOLD)))
MAX_EXTRA_FALSE_ALARMS = 10
MIN_STILL_FLAGGED = 43       # of the 53 originally flagged empty-profile frauds, under each injection
MAX_PR_AUC_DROP = 0.02
VAL_CSV = ROOT / "data" / "processed_v2" / "val.csv"
VAL_PROBABILITIES = ROOT / "backend" / "models_v2" / "distilbert_job_scam" / "val_probabilities.npy"
CACHE_PATH = ROOT / "reports" / "profile_runs" / "score_cache.json"
CHUNK = 64


class CachedScorer:
    """Wrap a scorer with an on-disk cache keyed by the text's SHA-256; saves after every chunk; reports progress."""

    def __init__(self, score, path: Path | None, log=None, chunk: int = CHUNK):
        self.score, self.path, self.log, self.chunk = score, path, log, chunk
        self.cache = {}
        if path is not None and path.exists():
            self.cache = json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def __call__(self, texts):
        pending = list(dict.fromkeys(t for t in texts if self.key(t) not in self.cache))
        started = time.time()
        for start in range(0, len(pending), self.chunk):
            part = pending[start:start + self.chunk]
            for text, value in zip(part, np.asarray(self.score(part), dtype=float)):
                self.cache[self.key(text)] = float(value)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(self.cache), encoding="utf-8")
            if self.log:
                done = min(start + self.chunk, len(pending))
                rate = (time.time() - started) / done
                self.log(f"[progress] {done}/{len(pending)} new texts scored, about {rate * (len(pending) - done) / 60:.0f} min left")
        return np.array([self.cache[self.key(t)] for t in texts])


def metrics(y: np.ndarray, logodds: np.ndarray) -> dict:
    flagged = logodds >= THRESHOLD_LOGODDS
    tp, fp = int((flagged & (y == 1)).sum()), int((flagged & (y == 0)).sum())
    fn = int(((~flagged) & (y == 1)).sum())
    return {"flagged": int(flagged.sum()), "tp": tp, "fp": fp, "fn": fn, "recall": tp / (tp + fn) if tp + fn else 0.0,
            "precision": tp / (tp + fp) if tp + fp else 0.0, "pr_auc": float(average_precision_score(y, logodds))}


def bootstrap_pr_auc_difference(y, base, other, groups, n_boot: int = N_BOOT) -> dict:
    index = list(pd.Series(np.arange(len(groups))).groupby(np.asarray(groups)).indices.values())
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(n_boot):
        picked = np.concatenate([index[i] for i in rng.integers(0, len(index), len(index))])
        if y[picked].min() == y[picked].max():
            continue
        diffs.append(average_precision_score(y[picked], other[picked]) - average_precision_score(y[picked], base[picked]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"difference": float(average_precision_score(y, other) - average_precision_score(y, base)),
            "ci95": [float(lo), float(hi)], "resamples": len(diffs)}


def defended_tier(lo_with: float, lo_stripped: float, has_profile: bool) -> str:
    """Tier under the defence: High only if the with-profile score flags it and a profile is present."""
    if lo_with >= THRESHOLD_LOGODDS:
        return VERDICT_HIGH_RISK if has_profile else VERDICT_NEEDS_VERIFICATION
    if max(lo_with, lo_stripped) >= THRESHOLD_LOGODDS:
        return VERDICT_NEEDS_VERIFICATION      # flagged only through the strip score
    return VERDICT_LEGITIMATE


def caught_under_defence(lo_with: float, lo_stripped: float, has_profile: bool, text: str) -> bool:
    tier = defended_tier(lo_with, lo_stripped, has_profile)
    return escalate_verdict(tier, "", detect_warning_signs(text))[0] != VERDICT_LEGITIMATE


def caught_deployed(lo_with: float, text: str, profile: str) -> bool:
    p = float(1 / (1 + np.exp(-lo_with)))
    verdict, reason = job_verdict(p, THRESHOLD, profile)
    return escalate_verdict(verdict, reason, detect_warning_signs(text))[0] != VERDICT_LEGITIMATE


def evaluate(val: pd.DataFrame, probabilities, score, n_boot: int = N_BOOT) -> dict:
    """score(texts) -> log-odds from the deployed model (a fake in tests)."""
    if len(val) != len(probabilities):
        raise ValueError("val.csv and val_probabilities.npy have different lengths")
    rows = val.to_dict("records")
    y = val[ds.LABEL_COL].to_numpy(dtype=int)
    groups = val[ds.GROUP_COL].to_numpy()
    baseline = logit(probabilities)
    profile = np.array([cleaned_profile(r) for r in rows])
    has_profile = profile != ""

    own_text = [text_with_profile(r, r.get("company_profile")) for r in rows]
    stripped_text = {i: text_with_profile(rows[i], None) for i in np.flatnonzero(has_profile)}
    empty_frauds = np.flatnonzero((y == 1) & ~has_profile)
    empty_legit = np.flatnonzero((y == 0) & ~has_profile)
    control = np.flatnonzero((y == 1) & has_profile)
    injected = {(pop, key): {i: text_with_profile(rows[i], text) for i in idx}
                for pop, idx in (("frauds", empty_frauds), ("legit", empty_legit)) for key, text in PROFILES.items()}
    control_text = {i: text_with_profile(rows[i], PROFILES["B"]) for i in control}

    needed = (own_text[:ALIGNMENT_ROWS] + list(stripped_text.values())
              + [t for d in injected.values() for t in d.values()] + list(control_text.values()))
    table = dict(zip(needed, np.asarray(score(needed), dtype=float)))

    probe = np.array([table[t] for t in own_text[:ALIGNMENT_ROWS]])
    gap = float(np.max(np.abs(probe - baseline[:len(probe)])))
    if gap > ALIGNMENT_TOLERANCE:
        raise RuntimeError(f"the scorer does not reproduce the saved validation log-odds (max gap {gap:.2e})")

    stripped = baseline.copy()                       # an empty profile is unchanged by definition
    for i, text in stripped_text.items():
        stripped[i] = table[text]
    defended = np.maximum(baseline, stripped)

    deployed_m, defended_m = metrics(y, baseline), metrics(y, defended)
    diff = bootstrap_pr_auc_difference(y, baseline, defended, groups, n_boot)

    originally_flagged = empty_frauds[baseline[empty_frauds] >= THRESHOLD_LOGODDS]
    injection = {"populations": {}, "control": {}, "still_flagged": {}}
    for pop, idx, name in (("frauds", empty_frauds, "frauds with no profile"), ("legit", empty_legit, "legitimate with no profile")):
        injection["populations"][name] = {}
        for key, text in PROFILES.items():
            inj = np.array([table[injected[(pop, key)][i]] for i in idx])
            after = np.maximum(inj, baseline[idx])        # profile removed = the original empty-profile text
            texts = [injected[(pop, key)][i] for i in idx]
            injection["populations"][name][key] = _row(idx, baseline[idx], after, inj, baseline[idx], texts, rows, groups, n_boot,
                                                       has_profile_after=True, original_profile=profile)
            if pop == "frauds":
                keep = np.isin(idx, originally_flagged)
                injection["still_flagged"][key] = int((after[keep] >= THRESHOLD_LOGODDS).sum())
    inj = np.array([table[control_text[i]] for i in control])
    after = np.maximum(inj, stripped[control])
    injection["control"]["B"] = _row(control, baseline[control], after, inj, stripped[control],
                                     [control_text[i] for i in control], rows, groups, n_boot, has_profile_after=True,
                                     original_profile=profile)

    def tier_counts(use_defence: bool) -> dict:
        counts = {t: {"n": 0, "fraud": 0} for t in (VERDICT_LEGITIMATE, VERDICT_NEEDS_VERIFICATION, VERDICT_HIGH_RISK)}
        for i in range(len(rows)):
            tier = defended_tier(baseline[i], stripped[i] if use_defence else baseline[i], bool(has_profile[i]))
            counts[tier]["n"] += 1
            counts[tier]["fraud"] += int(y[i])
        return counts

    result = {"alignment_gap": gap, "n": len(rows), "n_fraud": int(y.sum()), "deployed": deployed_m, "defended": defended_m,
              "pr_auc_difference": diff, "injection": injection,
              "tiers": {"deployed": tier_counts(False), "defended": tier_counts(True)}}
    result["conditions"] = conditions(result)
    return result


def _row(idx, before, after, with_profile, stripped, texts, rows, groups, n_boot, has_profile_after, original_profile) -> dict:
    lo, hi = bootstrap_mean(after - before, groups[idx], n_boot)
    caught_before = sum(caught_deployed(b, text_with_profile(rows[i], rows[i].get("company_profile")), original_profile[i])
                        for i, b in zip(idx, before))
    caught_after = sum(caught_under_defence(w, s, has_profile_after, t) for w, s, t in zip(with_profile, stripped, texts))
    return {"n": len(idx), "flagged_before": int((before >= THRESHOLD_LOGODDS).sum()),
            "flagged_after": int((after >= THRESHOLD_LOGODDS).sum()),
            "mean_shift": float((after - before).mean()) if len(idx) else float("nan"), "shift_ci95": [lo, hi],
            "caught_before": int(caught_before), "caught_after": int(caught_after)}


def conditions(result: dict) -> dict:
    deployed_fp, defended_fp = result["deployed"]["fp"], result["defended"]["fp"]
    still = result["injection"]["still_flagged"]
    drop = result["deployed"]["pr_auc"] - result["defended"]["pr_auc"]
    return {
        "1 false alarms": {"met": defended_fp - deployed_fp <= MAX_EXTRA_FALSE_ALARMS, "deployed": deployed_fp,
                           "defended": defended_fp, "rise": defended_fp - deployed_fp},
        "2 frauds still flagged": {"met": all(v >= MIN_STILL_FLAGGED for v in still.values()), "per_profile": dict(still)},
        "3 PR-AUC": {"met": drop <= MAX_PR_AUC_DROP, "deployed": result["deployed"]["pr_auc"],
                     "defended": result["defended"]["pr_auc"], "drop": drop},
    }


def _metrics_line(name: str, m: dict) -> str:
    return (f"  {name:<10}{m['flagged']:>8}{m['tp']:>6}{m['fp']:>6}{m['fn']:>6}{m['recall']:>9.3f}{m['precision']:>11.3f}{m['pr_auc']:>9.4f}")


def _row_line(label: str, s: dict, word: str) -> str:
    lo, hi = s["shift_ci95"]
    return (f"  {label:<4}{s['n']:>5}{s['flagged_before']:>9}{s['flagged_after']:>8}   {s['mean_shift']:>+8.3f}  "
            f"[{lo:+.3f}, {hi:+.3f}]   {s['caught_before']:>4} -> {s['caught_after']:<4} {word}")


def report_lines(result: dict) -> list[str]:
    d, f, diff, c = result["deployed"], result["defended"], result["pr_auc_difference"], result["conditions"]
    lines = [BANNER, f"threshold {THRESHOLD} (log-odds {THRESHOLD_LOGODDS:+.4f}); scorer max gap vs saved scores {result['alignment_gap']:.1e}",
             "", f"Full validation ({result['n']} postings, {result['n_fraud']} fraud)",
             f"  {'':<10}{'flagged':>8}{'TP':>6}{'FP':>6}{'FN':>6}{'recall':>9}{'precision':>11}{'PR-AUC':>9}",
             _metrics_line("deployed", d), _metrics_line("defended", f),
             f"  PR-AUC difference (defended - deployed): {diff['difference']:+.4f}  (95% group bootstrap {diff['ci95'][0]:+.4f} to "
             f"{diff['ci95'][1]:+.4f}, {diff['resamples']} resamples)", "", "Pre-registered adoption bar"]
    one, two, three = c["1 false alarms"], c["2 frauds still flagged"], c["3 PR-AUC"]
    mark = lambda ok: "MET" if ok else "NOT MET"  # noqa: E731
    lines += [f"  (1) false alarms rise <= {MAX_EXTRA_FALSE_ALARMS}: deployed {one['deployed']}, defended {one['defended']} "
              f"(rise {one['rise']:+d}) -> {mark(one['met'])}",
              f"  (2) >= {MIN_STILL_FLAGGED} of the originally flagged empty-profile frauds still flagged after each injection: "
              + ", ".join(f"({k}) {v}" for k, v in two["per_profile"].items()) + f" -> {mark(two['met'])}",
              f"  (3) PR-AUC not more than {MAX_PR_AUC_DROP} below deployed: {three['deployed']:.4f} -> {three['defended']:.4f} "
              f"(drop {three['drop']:+.4f}) -> {mark(three['met'])}"]
    header = f"  {'':<4}{'n':>5}{'flagged':>9}{'flagged':>8}   {'mean log-odds':>13}  {'95% group bootstrap':<20}  verdict Needs Verification or High Fraud Risk"
    sub = f"  {'':<4}{'':>5}{'deployed':>9}{'defended':>8}   {'shift':>13}"
    words = {"frauds with no profile": "caught", "legitimate with no profile": "false alarms"}
    for name, by_profile in result["injection"]["populations"].items():
        lines += ["", f"Injection under the defended score - {name}:", header, sub]
        lines += [_row_line(f"({k})", s, words[name]) for k, s in by_profile.items()]
    lines += ["", "Control under the defended score - frauds that already have a profile, profile replaced by (B):", header, sub]
    lines += [_row_line("(B)", s, "caught") for s in result["injection"]["control"].values()]
    lines += ["", "Tiers (counts and fraud counts)", f"  {'':<20}{'deployed':>18}{'defended':>18}"]
    for tier in (VERDICT_LEGITIMATE, VERDICT_NEEDS_VERIFICATION, VERDICT_HIGH_RISK):
        a, b = result["tiers"]["deployed"][tier], result["tiers"]["defended"][tier]
        lines.append(f"  {tier:<20}{str(a['n']) + ' (' + str(a['fraud']) + ' fraud)':>18}{str(b['n']) + ' (' + str(b['fraud']) + ' fraud)':>18}")
    high = result["tiers"]["defended"][VERDICT_HIGH_RISK]
    lines.append(f"  High Fraud Risk under the defence: {high['fraud']}/{high['n']}"
                 f" -> {'still all fraud' if high['n'] == high['fraud'] else 'no longer all fraud'}")
    return lines + ["", BANNER]


def main() -> int:
    from eval_input_formats_val import make_real_scorer

    log = lambda message: print(message, file=sys.stderr, flush=True)  # noqa: E731
    val = pd.read_csv(VAL_CSV)
    try:
        result = evaluate(val, np.load(VAL_PROBABILITIES), CachedScorer(make_real_scorer(), CACHE_PATH, log))
    except (RuntimeError, ValueError) as exc:
        print(f"Stopped: {exc}.", file=sys.stderr)
        return 3
    print("\n".join(report_lines(result)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
