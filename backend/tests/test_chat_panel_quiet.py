"""Calm chats: a clear line instead of an empty timeline list, and no guidance block. Display only."""
import shutil
import subprocess
from pathlib import Path

import pytest

from backend.services.conversation_analysis import analyze_messages
from backend.services.conversation_parser import parse_chat_text

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
APP = (FRONTEND / "src" / "App.jsx").read_text(encoding="utf-8")
TIMELINE = APP[APP.index("function Timeline("):APP.index("const CITATION")]
CHAT_PANEL = APP[APP.index('<h2>Conversation report</h2>'):APP.index("export default function App")]


def test_the_timeline_shows_the_line_and_no_empty_list_when_nothing_was_flagged():
    assert "{!items.some((m) => m.sign) && <p className=\"muted\">{NO_FLAGGED_MESSAGES}</p>}" in TIMELINE
    assert "{items.some((m) => m.sign) && (\n        <ul className=\"flags\">" in TIMELINE


def test_the_chat_panel_hides_guidance_for_a_calm_chat_and_the_job_report_is_untouched():
    assert "{showChatGuidance(report) && <GuidanceCards report={report} highlight={highlight} />}" in CHAT_PANEL
    job = (FRONTEND / "src" / "JobReport.jsx").read_text(encoding="utf-8")
    assert "<GuidanceCards report={report} highlight={null} />" in job and "showChatGuidance" not in job


def test_a_calm_chat_report_is_unchanged_by_the_display_rule():
    calm = analyze_messages(parse_chat_text("24/10/26, 9:00 - HR: Thanks for applying.\n24/10/26, 9:01 - You: When is my interview?\n"))
    assert (calm["risk_score"], calm["display_risk_score"], calm["risk_category"], calm["warning_signs"]) == (0, 1, "Lower risk", [])
    assert all(entry["sign"] is None for entry in calm["timeline"])


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_node_tests_for_the_chat_panel_pass():
    result = subprocess.run(["node", "--test", "src/services/chatPanel.test.mjs"], cwd=FRONTEND, capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
