"""Profile-injection measurement on a tiny synthetic frame with a fake scorer. No model, no real data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import eval_profile_injection_val as pi  # noqa: E402

CHEQUE = "We will mail you a cheque for $2,000. Deposit it and wire the balance to our supplier."
INJECTED_MARKERS = ("Company profile: x\n", "friendly team", "Northfield")


def fake_score(texts):
    """High log-odds for fee/cheque wording; any INJECTED profile pulls it down, an original profile does not."""
    out = []
    for text in texts:
        value = 4.0 if ("fee" in text or "cheque" in text) else -4.0
        if any(marker in text for marker in INJECTED_MARKERS):
            value -= 6.0
        out.append(value)
    return np.array(out)


def _rows():
    rows = []

    def add(label, profile, description, group):
        rows.append({"job_id": len(rows) + 1, "group_id": group, "fraudulent": label, "title": f"Job {len(rows)}",
                     "company_profile": profile, "description": description, "requirements": "None", "benefits": "Cash"})

    for i in range(4):
        add(1, None if i % 2 else "", "Pay a registration fee to start.", i)          # fraud, empty, one rule category
    add(1, "<p>&nbsp;</p>", CHEQUE, 10)                                                    # fraud, empty after cleaning, specific rule
    for i in range(5):
        add(0, "" if i % 2 else None, "Review reports with the team on site.", 20 + i)      # legitimate, empty
    add(0, "", "Pay a fee to the city council for your parking permit.", 30)               # legitimate, empty, flagged before
    for i in range(3):
        add(1, "Acme Ltd builds software for small clinics in the north.", "Pay a registration fee to start.", 40 + i)  # control
    add(0, "Acme Ltd builds software.", "Review reports with the team.", 50)
    return pd.DataFrame(rows)


def _probabilities(val):
    texts = [pi.text_with_profile(r, r["company_profile"]) for r in val.to_dict("records")]
    return 1 / (1 + np.exp(-fake_score(texts)))


@pytest.fixture(scope="module")
def result():
    val = _val = _rows()
    return pi.evaluate(val, _probabilities(val), fake_score, n_boot=200)


# --- the fixed inputs ----------------------------------------------------------------------------------------------

def test_the_three_injected_profiles_are_exactly_the_fixed_ones():
    assert pi.PROFILES["A"] == "x"
    assert pi.PROFILES["B"] == "We are a growing company with a friendly team. We value honesty and hard work."
    assert pi.PROFILES["C"] == (
        "Founded in 2005, Northfield Group is a established provider of business services to clients across several "
        "industries. Our team of experienced professionals is committed to quality, integrity and long-term "
        "relationships. We offer a supportive work environment, ongoing training and opportunities for career growth. "
        "Northfield Group is an equal opportunity employer.")
    assert list(pi.PROFILES) == ["A", "B", "C"] and pi.THRESHOLD == 0.291163 and pi.SEED == 42 and pi.N_BOOT == 2000


def test_text_with_profile_follows_the_structured_path():
    row = {"title": "Clerk", "company_profile": None, "description": "Enter data.", "requirements": None, "benefits": None}
    base = pi.text_with_profile(row, None)
    assert base == "Title: Clerk\nDescription: Enter data."
    assert pi.text_with_profile(row, "") == base == pi.text_with_profile(row, "<p>&nbsp;</p>")
    assert pi.text_with_profile(row, "x") == "Title: Clerk\nCompany profile: x\nDescription: Enter data."


def test_app_verdict_uses_the_apps_own_path_without_new_rules():
    assert pi.app_verdict(0.9, "Pay a fee.", "Acme") == "High Fraud Risk"
    assert pi.app_verdict(0.9, "Pay a fee.", "") == "Needs Verification"
    assert pi.app_verdict(0.01, "Review the reports.", "") == "Likely Legitimate"
    assert pi.app_verdict(0.01, CHEQUE, "x") == "Needs Verification"            # a specific rule category escalates alone
    assert pi.app_verdict(0.01, "You are hired without an interview. We need your online banking password.", "") == "Needs Verification"


# --- results on the synthetic frame --------------------------------------------------------------------------------

def test_injected_profiles_unflag_frauds_without_a_profile_and_the_shift_is_the_scorers_drop(result):
    frauds = result["populations"]["frauds with no profile"]
    for key in ("A", "B", "C"):
        s = frauds[key]
        assert (s["n"], s["flagged_before"], s["flagged_after"]) == (5, 5, 0), key
        assert s["mean_shift"] == pytest.approx(-6.0) and s["shift_ci95"] == [pytest.approx(-6.0), pytest.approx(-6.0)]


def test_frauds_still_end_as_a_caution_verdict_only_through_the_existing_rules(result):
    for key in ("A", "B", "C"):
        s = result["populations"]["frauds with no profile"][key]
        assert s["verdict_caught_before"] == 5     # all flagged by the classifier, empty profile -> Needs Verification
        assert s["verdict_caught_after"] == 1      # only the cheque posting, via its specific rule category


def test_false_alarm_side_for_legitimate_postings(result):
    for key in ("A", "B", "C"):
        s = result["populations"]["legitimate with no profile"][key]
        assert (s["n"], s["flagged_before"], s["flagged_after"]) == (6, 1, 0)
        assert s["verdict_caught_before"] == 1 and s["verdict_caught_after"] == 0   # false alarms before and after
        assert s["mean_shift"] == pytest.approx(-6.0)


def test_control_replaces_an_existing_profile_with_b(result):
    s = result["control"]["B"]
    assert (s["n"], s["flagged_before"], s["flagged_after"]) == (3, 3, 0)
    assert s["verdict_caught_before"] == 3 and s["verdict_caught_after"] == 0   # High Fraud Risk before, nothing after
    assert s["mean_shift"] == pytest.approx(-6.0)


# --- bootstrap, alignment, report ----------------------------------------------------------------------------------

def test_group_bootstrap_interval_brackets_the_mean_and_is_seeded():
    values = np.array([-1.0, -3.0, -2.0, -5.0, -4.0, -2.5, -3.5, -1.5])
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    lo, hi = pi.bootstrap_mean(values, groups, 500)
    assert lo <= values.mean() <= hi and lo < hi
    assert pi.bootstrap_mean(values, groups, 500) == (lo, hi)
    assert all(np.isnan(x) for x in pi.bootstrap_mean(np.array([]), np.array([]), 10))


def test_a_scorer_that_does_not_match_the_saved_scores_stops_the_run():
    val = _rows()
    with pytest.raises(RuntimeError, match="reproduce"):
        pi.evaluate(val, _probabilities(val) * 0.5, fake_score, n_boot=10)
    with pytest.raises(ValueError):
        pi.evaluate(val, np.zeros(3), fake_score, n_boot=10)


def test_report_has_the_banner_the_tables_and_the_control(result):
    lines = pi.report_lines(result)
    text = "\n".join(lines)
    assert lines[0] == pi.BANNER == lines[-1] == "VALIDATION ONLY - ADVERSARIAL PROFILE INJECTION. Not a test-set result."
    for part in ("frauds with no profile:", "legitimate with no profile:", "Control: frauds that already have a profile",
                 "(A) 1 word:", "(B) 15 words", "false alarms", "caught", "(A)", "(B)", "(C)"):
        assert part in text, part
    assert len([line for line in lines if line.startswith("  (")]) == 3 + 6 + 1  # profile list + 2 population tables of 3 + control


def test_the_script_is_validation_only_and_changes_nothing():
    source = (Path(__file__).resolve().parents[2] / "scripts" / "eval_profile_injection_val.py").read_text(encoding="utf-8")
    assert "test.csv" not in source.replace("never opens test.csv", "") and "heldout" not in source
    assert "client_from_env" not in source and "final_test_eval" not in source.replace("same structured path as final_test_eval", "")
