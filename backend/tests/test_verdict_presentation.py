"""Display-only wording for raised verdicts: the template explanation and the frontend line. No score changes."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from backend.services import job_analysis as ja
from backend.services import llm_explainer as lx
from backend.services.job_analysis import build_job_report

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
ESCALATING = "You are hired instantly. We need your online banking password and a selfie today."
CHEQUE_ONLY = "We will mail you a cheque for $2,000. Deposit it and wire the balance to our supplier."


def _prediction(probability, threshold=0.291163):
    return {"fraud_probability": probability, "threshold": threshold, "prediction": int(probability >= threshold),
            "fraud_logodds": -3.0, "model": "DistilBERT"}


def _template(report):
    facts = {k: report[k] for k in ("verdict", "verdict_reason", "display_risk_score")}
    return lx.template_explanation(facts, {})[0]


def test_two_category_reason_reads_without_nested_parentheses():
    facts = {"verdict": "Needs Verification", "display_risk_score": 17,
             "verdict_reason": "scam-pattern indicators present (Cheque or overpayment scam, No interview)"}
    text = lx.template_explanation(facts, {})[0]
    assert text.startswith("Verdict: Needs Verification. Reason: scam-pattern indicators present: "
                           "Cheque or overpayment scam, No interview. Risk score 17 out of 100")
    assert not re.search(r"\([^)]*\(", text) and "(" not in text.split("Risk score")[0]


def test_single_category_reason_reads_without_parentheses():
    facts = {"verdict": "Needs Verification", "display_risk_score": 15,
             "verdict_reason": "specific scam-pattern indicator present (Parcel or money forwarding)"}
    assert lx.template_explanation(facts, {})[0].startswith(
        "Verdict: Needs Verification. Reason: specific scam-pattern indicator present: Parcel or money forwarding. ")


@pytest.mark.parametrize("verdict, reason", [
    ("Likely Legitimate", "fraud probability is below the decision threshold"),
    ("Needs Verification", "classifier flagged the posting and the company profile is empty"),
    ("High Fraud Risk", "classifier flagged the posting even though a company profile is present"),
])
def test_other_reasons_keep_their_words(verdict, reason):
    text = lx.template_explanation({"verdict": verdict, "verdict_reason": reason, "display_risk_score": 50}, {})[0]
    assert text.startswith(f"Verdict: {verdict}. Reason: {reason}. ")


def test_chat_labels_still_work():
    facts = {"verdict": "Higher risk conversation", "verdict_reason": "Potentially fraudulent conversation",
             "display_risk_score": 80, "fact_label": "Assessment", "subject": "conversation"}
    assert lx.template_explanation(facts, {})[0].startswith(
        "Assessment: Higher risk conversation. Reason: Potentially fraudulent conversation. ")


def test_the_full_explanation_path_uses_the_new_wording_and_keeps_the_api_fields():
    report = build_job_report(_prediction(0.003), ESCALATING, "")
    assert report["verdict"] == "Needs Verification"
    assert report["verdict_reason"].startswith("scam-pattern indicators present (")  # API reason unchanged
    assert report["risk_category"] == "Lower risk"  # the field is not renamed or altered
    result = lx.explain({k: report[k] for k in ("verdict", "verdict_reason", "display_risk_score")}, {}, ESCALATING,
                        client=None, breaker=lx.CircuitBreaker())
    assert result["explanation_source"] == "template"
    assert result["explanation"].startswith("Verdict: Needs Verification. Reason: scam-pattern indicators present: ")
    assert "present (" not in result["explanation"]


def test_scores_and_category_are_identical_with_and_without_the_rule(monkeypatch):
    raised = build_job_report(_prediction(0.003), CHEQUE_ONLY, "")
    monkeypatch.setattr(ja, "escalate_verdict", lambda v, r, s: (v, r))
    plain = build_job_report(_prediction(0.003), CHEQUE_ONLY, "")
    assert raised["verdict"] != plain["verdict"]
    for key in ("risk_score", "risk_category", "display_risk_score", "classifier_risk_score", "warning_adjustment",
                "classifier_prediction", "classifier_details"):
        assert raised[key] == plain[key], key


def test_the_job_screen_leads_with_the_verdict_never_the_risk_category_and_keeps_the_note_in_the_details():
    job_report = (FRONTEND / "src" / "JobReport.jsx").read_text(encoding="utf-8")
    view = (FRONTEND / "src" / "services" / "reportView.js").read_text(encoding="utf-8")
    assert "risk_category" not in job_report  # the "Lower risk" chip is never shown for a job result
    assert "{screen.headline}" in job_report
    # the line about the verdict being raised by warning signs now sits in the collapsed technical section
    assert 'import { escalationNote } from "./verdictNote.js";' in view
    assert 'rows.push({ label: "Verdict note", text: note });' in view


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_node_tests_for_the_verdict_line_pass():
    result = subprocess.run(["node", "--test", "src/services/verdictNote.test.mjs"], cwd=FRONTEND,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
