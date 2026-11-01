"""Strip-score defence measurement on a tiny synthetic frame with a fake scorer. No model, no real data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import eval_profile_defense_val as pdv  # noqa: E402
import eval_profile_injection_val as inj  # noqa: E402

FEE = "Pay a registration fee to start."
COUNCIL = "Pay a fee to the city council for your parking permit."


def fake_score(texts):
    """+4 for fee wording, -4 otherwise; an injected generic profile or a 'Council' profile pulls it down by 6."""
    out = []
    for text in texts:
        value = 4.0 if ("fee" in text or "cheque" in text) else -4.0
        if "friendly team" in text or "Northfield" in text or "Company profile: x\n" in text or "Council profile" in text:
            value -= 6.0
        elif "ScamCo" in text:
            value += 1.0
        out.append(value)
    return np.array(out)


def _val():
    rows = []

    def add(label, profile, description, group):
        rows.append({"job_id": len(rows) + 1, "group_id": group, "fraudulent": label, "title": f"Job {len(rows)}",
                     "company_profile": profile, "description": description, "requirements": "None", "benefits": "Cash"})

    for i in range(4):
        add(1, "" if i % 2 else None, FEE, i)                                  # frauds, empty profile, flagged
    add(1, "", "Quietly pack boxes at home.", 4)                                # fraud, empty profile, not flagged
    for i in range(4):
        add(0, None, "Review reports with the team on site.", 10 + i)          # legitimate, empty profile
    add(0, "", COUNCIL, 20)                                                     # legitimate, empty, flagged (false alarm)
    add(1, "ScamCo trades goods.", FEE, 30)                                     # fraud with profile, flagged either way
    add(1, "Council profile of a generic firm.", FEE, 31)                       # fraud rescued by its profile
    add(0, "Council profile of a fine firm.", COUNCIL, 32)                      # legit rescued by its profile (new false alarm)
    add(0, "Council profile of a fine firm.", "Review reports with the team.", 33)
    return pd.DataFrame(rows)


def _probabilities(val):
    texts = [inj.text_with_profile(r, r["company_profile"]) for r in val.to_dict("records")]
    return 1 / (1 + np.exp(-fake_score(texts)))


@pytest.fixture(scope="module")
def result():
    val = _val()
    return pdv.evaluate(val, _probabilities(val), fake_score, n_boot=200)


# --- the defended score ---------------------------------------------------------------------------------------------

def test_full_validation_metrics_deployed_versus_defended(result):
    d, f = result["deployed"], result["defended"]
    assert (d["flagged"], d["tp"], d["fp"], d["fn"]) == (6, 5, 1, 2)
    assert (f["flagged"], f["tp"], f["fp"], f["fn"]) == (8, 6, 2, 1)   # +1 fraud recovered, +1 legitimate false alarm
    assert f["recall"] > d["recall"] and f["precision"] < d["precision"]
    assert result["n"] == 14 and result["n_fraud"] == 7


def test_defended_score_is_the_larger_of_the_two_and_empty_profiles_are_unchanged():
    val = _val()
    rows = val.to_dict("records")
    baseline = inj.logit(_probabilities(val))
    stripped = fake_score([inj.text_with_profile(r, None) for r in rows])
    defended = np.maximum(baseline, stripped)
    empty = np.array([inj.cleaned_profile(r) == "" for r in rows])
    assert np.allclose(defended[empty], baseline[empty])          # nothing changes without a profile
    assert (defended >= baseline).all() and defended[11] == pytest.approx(4.0) and baseline[11] == pytest.approx(-2.0)


def test_pr_auc_difference_has_an_interval_and_is_seeded(result):
    diff = result["pr_auc_difference"]
    assert diff["ci95"][0] <= diff["ci95"][1] and diff["resamples"] > 0
    assert diff["difference"] == pytest.approx(result["defended"]["pr_auc"] - result["deployed"]["pr_auc"])
    val = _val()
    y, base = val["fraudulent"].to_numpy(), inj.logit(_probabilities(val))
    again = pdv.bootstrap_pr_auc_difference(y, base, np.maximum(base, base + 0.5), val["group_id"].to_numpy(), 50)
    assert again == pdv.bootstrap_pr_auc_difference(y, base, np.maximum(base, base + 0.5), val["group_id"].to_numpy(), 50)


# --- the pre-registered bar --------------------------------------------------------------------------------------

def test_condition_one_counts_the_rise_in_false_alarms(result):
    one = result["conditions"]["1 false alarms"]
    assert (one["deployed"], one["defended"], one["rise"], one["met"]) == (1, 2, 1, True)


def test_condition_two_is_checked_per_injected_profile_and_needs_43(result, monkeypatch):
    two = result["conditions"]["2 frauds still flagged"]
    assert two["per_profile"] == {"A": 4, "B": 4, "C": 4}      # the 4 flagged empty-profile frauds stay flagged by construction
    assert two["met"] is False                                  # 4 < 43 on this tiny frame
    monkeypatch.setattr(pdv, "MIN_STILL_FLAGGED", 4)
    assert pdv.conditions(result)["2 frauds still flagged"]["met"] is True


def test_the_pre_registered_constants_are_fixed():
    assert (pdv.MAX_EXTRA_FALSE_ALARMS, pdv.MIN_STILL_FLAGGED, pdv.MAX_PR_AUC_DROP) == (10, 43, 0.02)
    assert pdv.THRESHOLD == 0.291163 and pdv.PROFILES is inj.PROFILES and list(pdv.PROFILES) == ["A", "B", "C"]


def test_condition_three_is_the_pr_auc_drop(result):
    three = result["conditions"]["3 PR-AUC"]
    assert three["drop"] == pytest.approx(three["deployed"] - three["defended"])
    assert three["met"] == (three["drop"] <= 0.02)


# --- injection under the defended score ----------------------------------------------------------------------------

def test_injection_into_flagged_frauds_cannot_unflag_them_under_the_defence(result):
    for key in ("A", "B", "C"):
        s = result["injection"]["populations"]["frauds with no profile"][key]
        assert (s["n"], s["flagged_before"], s["flagged_after"]) == (5, 4, 4)
        assert s["mean_shift"] == pytest.approx(0.0)
        assert (s["caught_before"], s["caught_after"]) == (4, 4)


def test_false_alarm_side_and_control(result):
    for key in ("A", "B", "C"):
        s = result["injection"]["populations"]["legitimate with no profile"][key]
        assert (s["n"], s["flagged_before"], s["flagged_after"]) == (5, 1, 1)
    control = result["injection"]["control"]["B"]
    assert (control["n"], control["flagged_before"], control["flagged_after"]) == (2, 1, 2)   # the rescued fraud is flagged too


# --- tiers --------------------------------------------------------------------------------------------------------

def test_tiers_high_needs_the_deployed_flag_and_a_profile_strip_only_flags_are_needs_verification(result):
    deployed, defended = result["tiers"]["deployed"], result["tiers"]["defended"]
    assert deployed["High Fraud Risk"] == {"n": 1, "fraud": 1} and defended["High Fraud Risk"] == {"n": 1, "fraud": 1}
    assert deployed["Needs Verification"] == {"n": 5, "fraud": 4}
    assert defended["Needs Verification"] == {"n": 7, "fraud": 5}      # + the rescued fraud and the rescued legitimate posting
    assert sum(t["n"] for t in defended.values()) == result["n"]
    assert pdv.defended_tier(1.0, -9.0, True) == "High Fraud Risk" and pdv.defended_tier(1.0, -9.0, False) == "Needs Verification"
    assert pdv.defended_tier(-3.0, 2.0, True) == "Needs Verification" and pdv.defended_tier(-3.0, -2.0, True) == "Likely Legitimate"


# --- cache, alignment, report -----------------------------------------------------------------------------------

def test_cache_scores_each_text_once_persists_and_reports_progress(tmp_path):
    calls, messages = [], []

    def counting(texts):
        calls.append(list(texts))
        return fake_score(texts)

    path = tmp_path / "cache" / "scores.json"
    scorer = pdv.CachedScorer(counting, path, messages.append, chunk=2)
    first = scorer(["fee a", "plain b", "fee a", "plain c"])
    assert list(first) == [4.0, -4.0, 4.0, -4.0] and sum(len(c) for c in calls) == 3     # duplicates scored once
    assert path.exists() and any("[progress]" in m for m in messages)
    again = pdv.CachedScorer(counting, path)(["fee a", "plain b"])
    assert list(again) == [4.0, -4.0] and sum(len(c) for c in calls) == 3                # served from the saved cache


def test_a_scorer_that_does_not_match_the_saved_scores_stops_the_run():
    val = _val()
    with pytest.raises(RuntimeError, match="reproduce"):
        pdv.evaluate(val, _probabilities(val) * 0.5, fake_score, n_boot=10)
    with pytest.raises(ValueError):
        pdv.evaluate(val, np.zeros(3), fake_score, n_boot=10)


def test_report_has_the_banner_the_bar_the_tables_and_the_tier_check(result):
    lines = pdv.report_lines(result)
    text = "\n".join(lines)
    assert lines[0] == pdv.BANNER == lines[-1] == "VALIDATION ONLY - PROFILE DEFENCE. Not a test-set result."
    for part in ("deployed", "defended", "PR-AUC difference (defended - deployed)", "Pre-registered adoption bar", "(1) false alarms",
                 "(2)", "(3) PR-AUC", "MET", "NOT MET", "frauds with no profile", "legitimate with no profile", "Control under the defended score",
                 "High Fraud Risk under the defence: 1/1 -> still all fraud"):
        assert part in text, part


def test_the_script_is_validation_only_and_changes_no_app_code():
    source = (Path(__file__).resolve().parents[2] / "scripts" / "eval_profile_defense_val.py").read_text(encoding="utf-8")
    assert "test.csv" not in source.replace("never opens test.csv", "") and "heldout" not in source
    assert "client_from_env" not in source and "job_routes" not in source
