"""'What to do next' lines for job and chat reports are display-only: no score or API field changes."""
import shutil
import subprocess
from pathlib import Path

import pytest

from backend.services.conversation_analysis import analyze_messages
from backend.services.conversation_parser import parse_chat_text
from backend.services.job_analysis import build_job_report

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
APP = (FRONTEND / "src" / "App.jsx").read_text(encoding="utf-8")
VIEW = (FRONTEND / "src" / "services" / "reportView.js").read_text(encoding="utf-8")
JOB_REPORT = (FRONTEND / "src" / "JobReport.jsx").read_text(encoding="utf-8")


def test_job_and_chat_reports_both_render_the_section_from_the_same_function():
    assert "nextStepsFor" in VIEW and "next: nextStepsFor(report)" in VIEW        # the job screen
    assert "const nextSteps = nextStepsFor(report);" in APP                        # the chat panel
    chat = APP[APP.index("const nextSteps = nextStepsFor(report);"):]
    assert "<h3>What to do next</h3>" in chat and "{nextSteps.map((line) => (" in chat
    assert "<h3>What to do next</h3>" in JOB_REPORT


def test_the_fixed_lines_live_in_one_place_and_the_cap_and_order_are_defined():
    for text in ("Do not deposit the cheque or send any of the money onward. Tell your bank.",
                 "Do not buy or send gift cards or crypto to anyone about a job.",
                 "Do not pay any fee to get a job, even if it is described as refundable.",
                 "If you have already paid or shared details: contact your bank at once. In India, report at cybercrime.gov.in or call 1930."):
        assert text in VIEW and text not in APP and text not in JOB_REPORT
    assert "export const MAX_ADVICE_LINES = 4;" in VIEW and "export const ADVICE_ORDER = Object.keys(CATEGORY_ADVICE);" in VIEW


def test_job_report_scores_and_fields_are_unchanged_by_the_new_lines():
    report = build_job_report({"fraud_probability": 0.05, "threshold": 0.291163, "prediction": 0, "fraud_logodds": -2.0, "model": "x"},
                              "We will mail you a cheque for $2,000. Deposit it and wire the balance to our supplier.", "")
    assert (report["risk_score"], report["display_risk_score"], report["classifier_risk_score"], report["warning_adjustment"]) == (15, 40, 5, 10)
    assert report["verdict"] == "Needs Verification" and report["risk_category"] == "Lower risk"
    assert set(report) == {"classifier_details", "classifier_prediction", "classifier_risk_score", "display_risk_score", "explanation",
                           "risk_category", "risk_score", "supporting_evidence", "uncertainty_notes", "verdict", "verdict_reason",
                           "warning_adjustment", "warning_signs"}


def test_chat_report_fields_and_scores_are_unchanged_by_the_new_lines():
    chat = analyze_messages(parse_chat_text(
        "24/10/26, 9:00 - Recruiter: Pay the registration fee of Rs 3000 within 2 hours. Send your bank details.\n"
        "24/10/26, 9:01 - You: Why?\n"))
    assert (chat["risk_score"], chat["display_risk_score"], chat["risk_category"]) == (80, 80, "Higher risk")
    assert chat["warning_signs"] == ["Payment request", "Sensitive information request", "Pressure tactic"]
    assert "next_steps" not in chat and "what_to_do" not in chat   # nothing was added to the API


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_node_tests_for_the_next_steps_pass():
    result = subprocess.run(["node", "--test", "src/services/nextSteps.test.mjs"], cwd=FRONTEND, capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
