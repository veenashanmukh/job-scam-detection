"""Pasted conversations: the opt-in parser fallback, the chat route, the routing code in the frontend, and the link
colours. Routing and display only: no rule, score, threshold or verdict logic changes."""
import io
import re
import shutil
import subprocess
from pathlib import Path

import pytest

flask = pytest.importorskip("flask")
from flask import Flask  # noqa: E402

from backend.routes.chat_routes import chat_bp  # noqa: E402
from backend.services.conversation_parser import parse_chat_text, parse_untimestamped_conversation  # noqa: E402

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
APP = (FRONTEND / "src" / "App.jsx").read_text(encoding="utf-8")
CSS = (FRONTEND / "src" / "styles.css").read_text(encoding="utf-8")

PLAIN = (
    "Recruiter: Hello, we found your profile and have a data entry role for you.\n"
    "You: Thanks, which company is this?\n"
    "Recruiter: Pay the registration fee of Rs 3000 within 2 hours.\n"
    "Recruiter: Send your bank details and add me on Telegram.\n"
)
TIMESTAMPED = (
    "24/10/26, 9:00 - Recruiter: Hello, we found your profile and have a data entry role for you.\n"
    "24/10/26, 9:01 - You: Thanks, which company is this?\n"
    "24/10/26, 9:02 - Recruiter: Pay the registration fee of Rs 3000 within 2 hours.\n"
    "24/10/26, 9:03 - Recruiter: Send your bank details and add me on Telegram.\n"
)


@pytest.fixture()
def client():
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(chat_bp)
    return app.test_client()


def _upload(client, raw, **form):
    return client.post("/api/analyze-chat", data={"file": (io.BytesIO(raw.encode()), "pasted_conversation.txt"), **form},
                       content_type="multipart/form-data")


# --- parser fallback ---------------------------------------------------------------------------------------------

def test_the_strict_parser_is_unchanged_and_still_refuses_untimestamped_text():
    assert parse_chat_text(PLAIN) == []
    assert len(parse_chat_text(TIMESTAMPED)) == 4


def test_untimestamped_fallback_reads_speaker_lines_with_empty_times_and_continuations():
    messages = parse_untimestamped_conversation(
        "Preamble line that is ignored\nRecruiter: Hello there.\nYou: Hi.\nHow are you?\nHR Desk: Fine, thanks.")
    assert [(m["speaker"], m["text"], m["time"], m["is_system"]) for m in messages] == [
        ("Recruiter", "Hello there.", "", False),
        ("You", "Hi.\nHow are you?", "", False),
        ("HR Desk", "Fine, thanks.", "", False),
    ]


def test_untimestamped_fallback_does_not_turn_urls_or_long_labels_into_speakers():
    messages = parse_untimestamped_conversation(
        "Recruiter: See this link.\nhttps://example.org/apply\nThis is far too long to be a speaker name at all: ok")
    assert [m["speaker"] for m in messages] == ["Recruiter"]
    assert "https://example.org/apply" in messages[0]["text"]
    with pytest.raises(TypeError):
        parse_untimestamped_conversation(None)


# --- route -------------------------------------------------------------------------------------------------------

def test_without_the_opt_in_an_untimestamped_paste_is_still_rejected(client):
    response = _upload(client, PLAIN)
    assert response.status_code == 400 and "timestamped" in response.get_json()["error"]


def test_with_the_opt_in_a_pasted_conversation_is_checked_as_a_chat(client):
    body = _upload(client, PLAIN, untimestamped="1").get_json()
    assert body["participants"] == ["Recruiter", "You"] and body["you"] == "You"
    assert {"Payment request", "Sensitive information request"} <= set(body["warning_signs"])
    assert all(entry["time"] == "" for entry in body["timeline"])


def test_a_real_export_is_parsed_strictly_whether_or_not_the_flag_is_sent(client):
    plain_flag = _upload(client, TIMESTAMPED).get_json()
    with_flag = _upload(client, TIMESTAMPED, untimestamped="1").get_json()
    assert plain_flag["timeline"][0]["time"] == "24/10/26 9:00" and with_flag["timeline"] == plain_flag["timeline"]
    assert with_flag["risk_score"] == plain_flag["risk_score"] and with_flag["risk_category"] == plain_flag["risk_category"]


def test_scores_for_the_same_conversation_are_identical_with_or_without_timestamps(client):
    pasted = _upload(client, PLAIN, untimestamped="1").get_json()
    exported = _upload(client, TIMESTAMPED).get_json()
    for key in ("risk_score", "risk_category", "warning_signs", "classifier_prediction", "participants", "you"):
        assert pasted[key] == exported[key], key
    assert [t["sign"] for t in pasted["timeline"]] == [t["sign"] for t in exported["timeline"]]
    assert [t["cumulative"] for t in pasted["timeline"]] == [t["cumulative"] for t in exported["timeline"]]


def test_a_paste_that_is_not_a_conversation_still_fails_cleanly_with_the_flag(client):
    response = _upload(client, "Just a job description with no speakers at all.", untimestamped="1")
    assert response.status_code == 400


# --- frontend routing code ---------------------------------------------------------------------------------------

def test_the_paste_box_routes_conversations_to_the_chat_analyzer_and_offers_the_override():
    assert 'if (routePastedText(text) === "chat") {' in APP
    assert "untimestamped: true" in APP and "routedFrom: { text, profile }" in APP
    assert "conversationFile(text)" in APP
    # the note and the override button appear on the routed bubble
    assert "{ROUTED_NOTE}" in APP and "{OVERRIDE_LABEL}" in APP and "onClick={() => checkAsJobPosting(m.routedFrom)}" in APP
    # the override goes straight to the job analyzer with the same text and profile, never back through routing
    override = APP[APP.index("function checkAsJobPosting"):]
    override = override[:override.index("\n  }\n") + 5]
    assert "runJob(routed.text, { text: routed.text, profile: routed.profile, userLabel: OVERRIDE_LABEL })" in override
    assert "routePastedText(text) ===" not in override and "runChat" not in override


def test_the_chat_call_passes_the_flag_and_a_recheck_keeps_the_routing():
    api = (FRONTEND / "src" / "services" / "api.js").read_text(encoding="utf-8")
    assert 'if (untimestamped) form.append("untimestamped", "1");' in api
    assert "analyzeChat(file, you, firstExplanation, untimestamped)" in APP
    assert "runChat(last.file, last.label, name, { untimestamped: last.untimestamped, routedFrom: last.routedFrom })" in APP


def test_messages_without_a_time_are_labelled_by_position_in_the_timeline():
    assert "const timeLabel = (item, index) => item.time || `message ${index + 1}`;" in APP
    assert "{items[0].time}" not in APP and "{m.time}" not in APP


def test_the_job_analyzer_path_is_untouched_for_ordinary_text():
    assert 'runJob(text, { text, profile: fields.company_profile });' in APP


# --- link colours ------------------------------------------------------------------------------------------------

def _variables(block):
    return dict(re.findall(r"--([a-z-]+):\s*(#[0-9a-fA-F]{6})", block))


def _luminance(color):
    channels = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    r, g, b = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a, b):
    high, low = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _themes():
    light_block = CSS[CSS.index(":root{"):CSS.index("@media (prefers-color-scheme: dark)")]
    dark_block = CSS[CSS.index("@media (prefers-color-scheme: dark)"):CSS.index("*{box-sizing")]
    light = _variables(light_block)
    return {"light": light, "dark": {**light, **_variables(dark_block)}}


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("link", ["link", "link-visited", "link-hover"])
def test_link_colours_have_enough_contrast_on_every_surface(theme, link):
    variables = _themes()[theme]
    for surface in ("card", "tint", "bg-a", "bg-b"):
        ratio = _contrast(variables[link], variables[surface])
        assert ratio >= 4.5, f"{theme} {link} on {surface}: {ratio:.2f}"


def test_the_stylesheet_styles_links_visited_hover_and_focus_with_the_theme_colours():
    assert "a{color:var(--link);" in CSS and "a:visited{color:var(--link-visited)}" in CSS
    assert "a:hover,a:focus-visible{color:var(--link-hover)}" in CSS
    assert "a:focus-visible{outline:2px solid var(--link)" in CSS
    assert ".link-button{" in CSS and "color:var(--link)" in CSS.split(".link-button{")[1].split("}")[0]


def test_the_dark_theme_links_are_not_the_unreadable_browser_default():
    dark = _themes()["dark"]
    assert _contrast("#0000ee", dark["card"]) < 2  # the default blue that was being shown
    assert _contrast(dark["link"], dark["card"]) > 7 and _contrast(dark["link-visited"], dark["card"]) > 7


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_node_tests_for_conversation_detection_pass():
    result = subprocess.run(["node", "--test", "src/services/conversation.test.mjs"], cwd=FRONTEND,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
