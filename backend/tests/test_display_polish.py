"""Display polish A-H. No score, threshold, rule, verdict logic, API field, model or checker change."""
import shutil
import subprocess
from pathlib import Path

import pytest

from backend.services.job_analysis import build_job_report

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
SRC = FRONTEND / "src"
APP = (SRC / "App.jsx").read_text(encoding="utf-8")
VIEW = (SRC / "services" / "reportView.js").read_text(encoding="utf-8")
API = (SRC / "services" / "api.js").read_text(encoding="utf-8")
CSS = (SRC / "styles.css").read_text(encoding="utf-8")
JOB = (SRC / "JobReport.jsx").read_text(encoding="utf-8")


def test_a_raw_probability_replaces_the_classifier_score_line():
    assert 'label: "Raw classifier probability"' in VIEW and "Score details" not in VIEW
    assert '" (saturated)"' in VIEW and "p > 0.99 || p < 0.01" in VIEW
    assert "classifier_risk_score" not in APP and "classifier_risk_score" not in JOB


def test_a_scores_are_clamped_everywhere_they_are_shown():
    assert "clampDisplayScore(report.display_risk_score)" in VIEW and "clampDisplayScore(report.risk_score)" in VIEW
    assert "Risk score {clampDisplayScore(m.report.display_risk_score)} / 100." in APP


def test_b_chat_label_is_rule_check():
    assert "<span>Rule check:</span>" in APP and "<span>Classifier:</span>" not in APP


def test_c_rules_table_scrolls_and_the_id_column_never_wraps():
    assert '<div className="table-scroll">' in JOB and '<table className="rules-table">' in JOB
    assert ".table-scroll{max-width:100%;overflow-x:auto}" in CSS
    assert ".rules-table th:first-child,.rules-table td:first-child{min-width:3.2em;white-space:nowrap}" in CSS


def test_d_bare_failures_show_the_server_unreachable_message():
    assert 'export const SERVER_UNREACHABLE = "Could not reach the analysis server. Please try again in a moment.";' in API
    assert API.count("await send(") == 3 and "failed (${res.status})" not in API


def test_e_f_g_h_are_in_place():
    assert (Path(__file__).resolve().parents[1] / "services" / "text_display.py").exists()           # E: word-boundary quotes
    assert "plainTags(e.warning_sign_tags)" in (SRC / "components" / "GuidanceCards.jsx").read_text(encoding="utf-8")  # F
    assert "{NO_FLAGGED_MESSAGES}" in APP                                                             # G
    assert "{showChatGuidance(report) && <GuidanceCards" in APP                                       # H


def test_no_score_or_api_field_changed():
    report = build_job_report({"fraud_probability": 0.999614, "threshold": 0.291163, "prediction": 1, "fraud_logodds": 7.86,
                               "model": "DistilBERT"}, "Pay a registration fee today.", "")
    assert report["classifier_risk_score"] == 100 and report["display_risk_score"] == 94   # API values as before
    assert report["classifier_details"]["fraud_probability"] == 0.999614


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_node_tests_for_the_polish_pass():
    result = subprocess.run(["node", "--test", "src/services/polish.test.mjs", "src/services/jobReportRender.test.mjs"],
                            cwd=FRONTEND, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
