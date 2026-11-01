"""eval_input_formats_val and the --single-category measurement, on tiny synthetic data with a fake scorer."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import eval_escalation_val as ee  # noqa: E402
import eval_input_formats_val as fv  # noqa: E402

CHEQUE_ONLY = "We will mail you a cheque for $2,000. Deposit it and wire the balance to our supplier."
FORWARD_ONLY = "Receive parcels at your home address and forward them to our clients abroad."
CHEQUE_AND_NO_INTERVIEW = CHEQUE_ONLY + " No experience or interview required."
PLAIN_JOB = "Build and review software with the platform team."


def fake_score(texts):
    """Log-odds that depend on the words, and a little on the labels so formats differ."""
    return np.array([(2.0 if ("cheque" in t or "parcels" in t) else -2.0) + (0.3 if t.startswith("Title:") else 0.0)
                     for t in texts])


def _val(n_legit=30):
    rows = []
    for i in range(6):  # frauds
        rows.append({"job_id": i, "group_id": i // 2, "fraudulent": 1, "title": f"Fraud job {i}", "company_profile": None,
                     "description": CHEQUE_ONLY, "requirements": "bank account", "benefits": "free laptop"})
    for i in range(n_legit):
        rows.append({"job_id": 100 + i, "group_id": 100 + i // 2, "fraudulent": 0, "title": f"Job {i}",
                     "company_profile": "A Firm", "description": PLAIN_JOB, "requirements": "degree", "benefits": "holiday"})
    frame = pd.DataFrame(rows)
    frame["text"] = [fv.format_texts(r)["structured"] for r in frame.to_dict("records")]
    return frame


def _probs(val):
    return 1 / (1 + np.exp(-fake_score(val["text"].tolist())))


def test_sample_has_every_fraud_and_a_seeded_legit_subset():
    val = _val()
    sample = fv.sample_rows(val, 10)
    assert int(sample["fraudulent"].sum()) == 6 and (sample["fraudulent"] == 0).sum() == 10
    assert sample.index.tolist() == sorted(sample.index.tolist())
    assert fv.sample_rows(val, 10).index.tolist() == sample.index.tolist()  # deterministic
    assert len(fv.sample_rows(val, 999)) == len(val)  # capped at what exists


def test_three_formats_are_built_as_described():
    row = {"title": "<b>Clerk</b>", "company_profile": "", "description": "Do tasks.  Earn well.",
           "requirements": "Phone", "benefits": None}
    texts = fv.format_texts(row)
    assert texts["structured"] == "Title: Clerk\nDescription: Do tasks. Earn well.\nRequirements: Phone"
    assert texts["plain"] == "Clerk\nDo tasks. Earn well.\nPhone"  # same cleaned fields, no labels, no empty fields
    # build_text collapses whitespace inside a field, so the labelled version is a single line
    assert texts["description-only"] == "Description: Clerk Do tasks. Earn well. Phone"


def test_metrics_at_the_threshold():
    y = np.array([1, 1, 0, 0])
    m = fv.format_metrics(y, np.array([2.0, -3.0, 1.0, -2.0]))  # flagged: first and third
    assert (m["tp"], m["false_positives"], m["n_flagged"]) == (1, 1, 2)
    assert m["recall"] == 0.5 and m["precision"] == 0.5 and m["pr_auc"] == pytest.approx(0.75)


def test_bootstrap_difference_is_zero_for_identical_scores_and_positive_for_a_better_format():
    rng = np.random.default_rng(0)
    y = np.array([1] * 30 + [0] * 70)
    groups = np.arange(100) // 2
    good = np.where(y == 1, 2.0, -2.0) + rng.normal(0, 1, 100)
    assert fv.bootstrap_pr_auc_difference(y, good, good, groups, 50)["ci95"] == [0.0, 0.0]
    noise = rng.normal(0, 1, 100)
    diff = fv.bootstrap_pr_auc_difference(y, noise, good, groups, 200)
    assert diff["difference"] > 0 and diff["ci95"][0] > 0 and diff["resamples_used"] + diff["skipped"] == 200


def test_evaluate_reports_every_format_and_checks_alignment():
    val = _val()
    result = fv.evaluate(val, _probs(val), fake_score, n_legit=20, n_boot=30)
    assert result["n"] == 26 and result["n_fraud"] == 6 and result["alignment_gap"] < 1e-9
    assert set(result["formats"]) == set(fv.FORMATS) and "pr_auc_difference_vs_structured" not in result["formats"]["structured"]
    for name in ("plain", "description-only"):
        assert set(result["formats"][name]["pr_auc_difference_vs_structured"]) >= {"difference", "ci95"}
    assert result["mean_logodds_shift_vs_structured"]["plain"] == pytest.approx(-0.3)  # no "Title:" bonus in plain
    with pytest.raises(RuntimeError, match="reproduce"):
        fv.evaluate(val, _probs(val) * 0.5, fake_score, 20, 10)
    with pytest.raises(ValueError):
        fv.evaluate(val, _probs(val)[:-1], fake_score, 20, 10)


def test_cli_banner_limits_and_failure_codes(capsys):
    val = _val()
    assert fv.main(["--n-legit", "20", "--n-boot", "20"], score=fake_score, val=val, probabilities=_probs(val)) == 0
    out = capsys.readouterr().out
    assert out.count("VALIDATION ONLY") == 2 and "structured" in out and "plain" in out and "description-only" in out
    assert fv.main(["--n-legit", "401"], score=fake_score, val=val, probabilities=_probs(val)) == 2
    assert fv.main(["--n-legit", "20"], score=fake_score, val=val, probabilities=_probs(val) * 0.5) == 3


# --- --single-category -------------------------------------------------------------------------------

def _escalation_val():
    rows = []
    for label, text in [(1, CHEQUE_ONLY), (0, CHEQUE_ONLY), (0, CHEQUE_ONLY), (1, FORWARD_ONLY), (0, PLAIN_JOB),
                        (1, CHEQUE_AND_NO_INTERVIEW), (1, CHEQUE_ONLY)]:
        rows.append({"job_id": len(rows), "fraudulent": label, "text": text, "company_profile": ""})
    return pd.DataFrame(rows)


def test_single_category_counts_only_likely_legitimate_postings_where_the_label_fires_alone():
    val = _escalation_val()
    probabilities = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.9])  # last one is flagged by the classifier
    result = ee.compute(val, probabilities, 0.291163)
    single = result["single_category"]
    assert single["labels"]["Cheque or overpayment scam"] == {"legit": 2, "fraud": 1}
    assert single["labels"]["Parcel or money forwarding"] == {"legit": 0, "fraud": 1}
    assert single["labels"]["Gift card or crypto payment"] == {"legit": 0, "fraud": 0}
    assert (single["legit_total"], single["fraud_total"], single["bar_max_false_alarms"]) == (2, 2, 10)
    # escalated now: the three lone cheque/forwarding postings and the two-category one (the flagged one is not touched)
    assert result["changed"] == 5


def test_single_category_report_lines_and_flag(monkeypatch, capsys):
    val = _escalation_val()
    probabilities = np.full(len(val), 0.05)
    result = ee.compute(val, probabilities, 0.291163)
    text = "\n".join(ee.single_category_lines(result))
    assert "Pre-registered bar" in text and "<= 10" in text and "observed 2" in text
    monkeypatch.setattr(ee.pd, "read_csv", lambda *_a, **_k: val)
    monkeypatch.setattr(ee.np, "load", lambda *_a, **_k: probabilities)
    assert ee.main(["--single-category"]) == 0
    with_flag = capsys.readouterr().out
    assert "Single-category measurement" in with_flag and with_flag.rstrip().endswith(ee.BANNER)
    assert ee.main([]) == 0 and "Single-category measurement" not in capsys.readouterr().out
