"""eval_synthetic_jobs and eval_escalation_val on tiny temp data with a fake classifier (never real data)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import eval_escalation_val as ee  # noqa: E402
import eval_synthetic_jobs as sj  # noqa: E402

THRESHOLD = 0.291163
SCAM_TWO = {"title": "Clerk", "company_profile": "", "requirements": "None",
            "benefits": "Pay weekly", "description": "You are hired instantly. We need your online banking password today."}
SCAM_ONE = {"title": "Packer", "company_profile": "", "requirements": "None", "benefits": "",
            "description": "Receive parcels at your home address and forward them to our clients abroad."}
BENIGN_ONE = {"title": "Analyst", "company_profile": "A Firm", "requirements": "Degree", "benefits": "Holiday",
              "description": "Interviews run on video. No experience or interview required for the optional workshop."}
BENIGN_NONE = {"title": "Engineer", "company_profile": "A Firm", "requirements": "Degree", "benefits": "Holiday",
               "description": "Build and review software with the platform team."}


def fake_predict(text):
    return {"fraud_probability": 0.05, "fraud_logodds": -2.9, "threshold": THRESHOLD}


def _write_postings(tmp_path, rows):
    postings = [{"id": f"p{i}", "label": label, "difficulty": "easy", "type": f"t{i}", **fields}
                for i, (label, fields) in enumerate(rows)]
    (tmp_path / "postings.json").write_text(json.dumps(postings), encoding="utf-8")
    return postings


ROWS = [("scam", SCAM_TWO), ("scam", SCAM_ONE), ("benign", BENIGN_ONE), ("benign", BENIGN_NONE)]


def test_synthetic_jobs_metrics_with_and_without_escalation(tmp_path):
    rows = sj.score_postings(_write_postings(tmp_path, ROWS), fake_predict)
    by_id = {r["id"]: r for r in rows}
    assert by_id["p0"]["verdict_plain"] == "Likely Legitimate" and by_id["p0"]["verdict_escalated"] == "Needs Verification"
    assert len(by_id["p0"]["categories"]) >= 2 and by_id["p0"]["escalation_reason"].startswith("scam-pattern indicators")
    assert by_id["p1"]["verdict_escalated"] == "Needs Verification"  # a specific category (forwarding) escalates alone
    assert by_id["p2"]["verdict_escalated"] == "Likely Legitimate" and by_id["p3"]["categories"] == []
    plain, escalated = sj.metrics(rows, "verdict_plain"), sj.metrics(rows, "verdict_escalated")
    assert (plain["tp"], plain["fp"], plain["recall"]) == (0, 0, 0.0)
    assert (escalated["tp"], escalated["fp"], escalated["fn"], escalated["tn"]) == (2, 0, 0, 2)
    assert escalated["precision"] == 1.0 and escalated["recall"] == 1.0 and escalated["accuracy"] == 1.0
    assert by_id["p1"]["escalation_reason"] == "specific scam-pattern indicator present (Parcel or money forwarding)"


def test_synthetic_jobs_report_and_cli(tmp_path, capsys):
    _write_postings(tmp_path, ROWS)
    assert sj.main(["--data-dir", str(tmp_path)], predict=fake_predict) == 0
    out = capsys.readouterr().out
    assert sj.BANNER in out and str(tmp_path) in out and "2 scam, 2 benign" in out
    assert "without escalation" in out and "with escalation" in out and "Misclassified with escalation: 0" in out and "Misclassified without escalation: 2" in out


def test_synthetic_jobs_needs_both_labels(tmp_path, capsys):
    _write_postings(tmp_path, [("scam", SCAM_TWO), ("scam", SCAM_ONE)])
    assert sj.main(["--data-dir", str(tmp_path)], predict=fake_predict) == 2
    assert "at least 1 scam and 1 benign" in capsys.readouterr().err
    assert sj.main(["--data-dir", str(tmp_path / "missing")], predict=fake_predict) == 2


def _val_frame():
    rows = []
    for label, fields in [(1, SCAM_TWO), (0, SCAM_TWO), (1, SCAM_ONE), (0, BENIGN_ONE), (0, BENIGN_NONE), (1, BENIGN_NONE)]:
        text = f"Title: {fields['title']}\nDescription: {fields['description']}"
        rows.append({"job_id": len(rows), "fraudulent": label, "text": text, "company_profile": fields["company_profile"]})
    return pd.DataFrame(rows)


def test_escalation_val_counts_changes_by_label_and_the_tier_table():
    val = _val_frame()
    probabilities = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.9])  # last posting is flagged: not Likely Legitimate
    result = ee.compute(val, probabilities, THRESHOLD)
    # two postings with two categories (one fraud, one legitimate) and one fraud with a lone specific category
    assert (result["changed"], result["changed_fraud"], result["changed_legit"]) == (3, 2, 1)
    assert result["likely_legitimate_before"] == 5 and result["fraud_in_tier"] == 2 and result["legit_in_tier"] == 3
    tiers = {t["tier"]: t for t in result["tiers"]}
    assert tiers["Likely Legitimate"]["before"] == {"n": 5, "fraud": 2} and tiers["Likely Legitimate"]["after"] == {"n": 2, "fraud": 0}
    assert tiers["Needs Verification"]["after"]["n"] == tiers["Needs Verification"]["before"]["n"] + 3
    assert sum(t["after"]["n"] for t in result["tiers"]) == sum(t["before"]["n"] for t in result["tiers"]) == 6
    assert sum(result["category_combinations"].values()) == 3


def test_escalation_val_never_lowers_a_verdict_and_reports_a_banner():
    val = _val_frame()
    result = ee.compute(val, np.full(len(val), 0.9), THRESHOLD)  # everything flagged by the classifier
    assert result["changed"] == 0
    lines = ee.report_lines(result)
    assert lines[0] == ee.BANNER and lines[-1] == ee.BANNER and any("verdict tier" in line for line in lines)


def test_escalation_val_rejects_misaligned_probabilities():
    with pytest.raises(ValueError):
        ee.compute(_val_frame(), np.zeros(3), THRESHOLD)
