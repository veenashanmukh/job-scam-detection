"""Highlighted phrases must belong to a listed warning sign (frontend/src/services/highlight.js)."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from backend.services.job_analysis import detect_warning_signs

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
POSTING = ("Earn Rs 90,000 per month from home. Contact our manager on Telegram when convenient, "
           "a deposit is not required and there is no registration fee.")


def test_job_rules_list_only_the_salary_sign_for_a_telegram_posting():
    signs = detect_warning_signs(POSTING)
    assert [s["label"] for s in signs] == ["Unusually high salary claim"]  # no Telegram rule exists for postings
    assert all(s["matched_text"] in POSTING for s in signs)  # each phrase is really in the text, so it can be marked


def test_every_sign_phrase_occurs_in_the_text_it_came_from():
    text = "Pay a registration fee today. Send your bank details to start. Guaranteed placement, no interview."
    signs = detect_warning_signs(text)
    assert len(signs) >= 3
    for sign in signs:
        assert re.search(re.escape(sign["matched_text"]), text, re.IGNORECASE)


def test_the_frontend_has_no_hard_coded_highlight_word_list():
    app = (FRONTEND / "src" / "App.jsx").read_text(encoding="utf-8")
    job_report = (FRONTEND / "src" / "JobReport.jsx").read_text(encoding="utf-8")  # the highlighter moved here
    module = (FRONTEND / "src" / "services" / "highlight.js").read_text(encoding="utf-8").lower()
    assert "const HL" not in app and "const HL" not in job_report
    assert "highlightSegments(text, signs)" in job_report and "signs={report.warning_signs}" in job_report
    for word in ("telegram", "deposit", "urgent", "aadhaar", "guaranteed"):
        assert word not in module


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_highlight_spans_are_a_subset_of_the_listed_signs_matched_text():
    result = subprocess.run(["node", "--test", "src/services/highlight.test.mjs"], cwd=FRONTEND,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
