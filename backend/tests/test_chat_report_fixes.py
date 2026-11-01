"""Chat report fixes: one clamp helper for the shown score, the default "You", and the wording moves.
No chat rule, rule weight or job-report behaviour changes."""
import io
import math
import shutil
import subprocess
from pathlib import Path

import pytest

flask = pytest.importorskip("flask")
from flask import Flask  # noqa: E402

from backend.routes import chat_routes  # noqa: E402
from backend.routes.chat_routes import chat_bp  # noqa: E402
from backend.services.conversation_analysis import analyze_messages, detect_self  # noqa: E402
from backend.services.conversation_parser import parse_chat_text  # noqa: E402
from backend.services.job_analysis import display_risk_score  # noqa: E402
from backend.services.score_display import clamp_display_score  # noqa: E402

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
APP = (FRONTEND / "src" / "App.jsx").read_text(encoding="utf-8")
CHAT_VIEW = (FRONTEND / "src" / "services" / "chatView.js").read_text(encoding="utf-8")

BENIGN = "24/10/26, 9:00 - HR: Thanks for applying.\n24/10/26, 9:01 - You: When is my interview?\n"
SCAM = (
    "24/10/26, 9:00 - Recruiter: Pay the registration fee of Rs 3000 within 2 hours. Send your bank details "
    "and add me on Telegram. Guaranteed job, no interview.\n24/10/26, 9:01 - You: Why?\n"
)


@pytest.fixture()
def client():
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(chat_bp)
    return app.test_client()


def _post(client, raw, **form):
    return client.post("/api/analyze-chat", data={"file": (io.BytesIO(raw.encode()), "chat.txt"), **form},
                       content_type="multipart/form-data").get_json()


# --- the shared clamp ---------------------------------------------------------------------------------------------

def test_clamp_is_one_to_ninety_nine_and_monotone():
    assert [clamp_display_score(v) for v in (-5, 0, 0.4, 1, 50, 99, 99.9, 100, 250)] == [1, 1, 1, 1, 50, 99, 99, 99, 99]
    shown = [clamp_display_score(v) for v in range(101)]
    assert shown == sorted(shown) and all(1 <= s <= 99 for s in shown)
    assert [clamp_display_score(v) for v in range(1, 100)] == list(range(1, 100))  # distinct in-range scores stay distinct


def test_job_display_score_is_unchanged_by_sharing_the_clamp():
    def old(logodds, threshold):  # the formula as it was before the helper existed
        z = (float(logodds) - math.log(threshold / (1.0 - threshold))) / 3.0
        s = 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))
        return int(min(99.0, max(1.0, 100.0 * s)))

    for threshold in (0.05, 0.291163, 0.5, 0.9):
        for step in range(-400, 401):
            logodds = step / 20.0
            assert display_risk_score(logodds, threshold) == old(logodds, threshold), (logodds, threshold)
    assert display_risk_score(None, 0.29) is None


# --- chat display score -------------------------------------------------------------------------------------------

def test_chat_raw_zero_shows_one_and_raw_hundred_shows_ninety_nine_with_existing_fields_unchanged():
    low = analyze_messages(parse_chat_text(BENIGN))
    high = analyze_messages(parse_chat_text(SCAM))
    assert (low["risk_score"], low["display_risk_score"]) == (0, 1)
    assert (high["risk_score"], high["display_risk_score"]) == (100, 99)
    assert low["risk_category"] == "Lower risk" and high["risk_category"] == "Higher risk"


def test_chat_display_scores_keep_the_order_of_the_raw_scores():
    chats = [BENIGN, "24/10/26, 9:00 - Recruiter: Move to Telegram please.\n24/10/26, 9:01 - You: Ok\n",
             "24/10/26, 9:00 - Recruiter: Pay the registration fee of Rs 3000 now.\n24/10/26, 9:01 - You: Ok\n", SCAM]
    reports = [analyze_messages(parse_chat_text(c)) for c in chats]
    raw = [r["risk_score"] for r in reports]
    shown = [r["display_risk_score"] for r in reports]
    assert raw == sorted(raw) and len(set(raw)) == 4
    assert shown == sorted(shown) and all(1 <= s <= 99 for s in shown)


def test_chat_route_returns_the_new_field_and_the_explanation_uses_it(client):
    body = _post(client, BENIGN)
    assert body["risk_score"] == 0 and body["display_risk_score"] == 1
    assert "Risk score 1 out of 100" in body["explanation"] and "Risk score 0 out of 100" not in body["explanation"]
    high = _post(client, SCAM)
    assert high["risk_score"] == 100 and high["display_risk_score"] == 99
    assert "display_risk_score" in chat_routes.FIXED_CHAT_KEYS


# --- default "You" ------------------------------------------------------------------------------------------------

C1 = "Recruiter: Hello.\nYou: Hi.\nRecruiter: Pay the fee.\n"
C2 = ("[24/10/26, 9:02 AM] Recruiter: Hello.\n[24/10/26, 9:03 AM] You: Hi.\n[24/10/26, 9:05 AM] Recruiter: Pay the fee.\n")
C3 = ("24/10/26, 9:00 - HR Desk: Pay the fee.\n24/10/26, 9:01 - You: Why?\n24/10/26, 9:02 - HR Desk: Send documents.\n")


def _pasted(raw):
    messages = parse_chat_text(raw)
    if not messages:
        from backend.services.conversation_parser import parse_untimestamped_conversation
        messages = parse_untimestamped_conversation(raw)
    return messages


@pytest.mark.parametrize("raw", [C1, C2, C3])
def test_c1_c2_c3_default_to_you(raw):
    assert detect_self(_pasted(raw)) == ("You", "export_label")
    assert analyze_messages(_pasted(raw))["you"] == "You"


@pytest.mark.parametrize("label", ["Me", "Myself", "Candidate", "Applicant", "Seeker", "you"])
def test_self_labels_are_preselected(label):
    messages = parse_chat_text(f"24/10/26, 9:00 - Priya: Hello\n24/10/26, 9:01 - {label}: Hi\n24/10/26, 9:02 - Priya: Pay now\n")
    assert detect_self(messages) == (label, "export_label")


@pytest.mark.parametrize("recruiter", ["Recruiter", "HR", "HR Desk", "Hiring Manager", "Agent", "Employer", "Manager"])
def test_the_speaker_who_is_not_on_the_recruiter_side_is_preselected(recruiter):
    messages = parse_chat_text(f"24/10/26, 9:00 - {recruiter}: Hello\n24/10/26, 9:01 - Priya: Hi\n24/10/26, 9:02 - {recruiter}: Pay now\n")
    assert detect_self(messages) == ("Priya", "not_recruiter")


@pytest.mark.parametrize("people", [["Priya", "Sam"], ["Recruiter", "HR"], ["Recruiter", "Priya", "Sam"], ["Priya", "Sam", "Jo"]])
def test_ambiguous_chats_leave_the_choice_empty_instead_of_guessing(people):
    lines = [f"24/10/26, 9:{i:02d} - {p}: Can I ask something? I wonder about my pay." for i, p in enumerate(people)]
    lines += [f"24/10/26, 9:{len(people):02d} - {people[0]}: Hello again."]
    result = analyze_messages(parse_chat_text("\n".join(lines) + "\n"))
    assert result["you"] is None and result["you_source"] == "undetermined"
    assert "not clear which participant is you" in " ".join(result["uncertainty_notes"])


def test_a_chat_with_two_names_never_picks_the_first_speaker_or_the_one_asking_questions():
    raw = "24/10/26, 9:00 - Priya: Can I ask about my pay? May I know the company?\n24/10/26, 9:01 - Sam: Yes.\n"
    assert detect_self(parse_chat_text(raw)) == (None, "undetermined")
    reversed_raw = "24/10/26, 9:00 - Sam: Yes.\n24/10/26, 9:01 - Priya: Can I ask about my pay?\n"
    assert detect_self(parse_chat_text(reversed_raw)) == (None, "undetermined")


def test_a_chosen_name_still_wins_and_two_self_labels_are_ambiguous():
    assert detect_self(parse_chat_text("24/10/26, 9:00 - Recruiter: Hi\n24/10/26, 9:01 - You: Hi\n"), self_name="Recruiter") == ("Recruiter", "selected")
    both = "24/10/26, 9:00 - You: Hi\n24/10/26, 9:01 - Me: Hello\n"
    assert detect_self(parse_chat_text(both)) == (None, "undetermined")


# --- wording moves in the frontend ---------------------------------------------------------------------------------

def test_the_chat_panel_and_bubble_show_the_new_score_and_wording():
    assert "Provisional score. It is not a probability." not in APP and "Provisional" not in APP
    assert '<p className="hint">Higher means more risk</p>' in APP
    assert "score={chatScore(report)}" in APP and "Risk score {chatScore(m.report)} / 100." in APP
    assert "Score {m.report.risk_score}" not in APP
    assert "risk_score" not in APP.replace("display_risk_score", "")  # the raw score is no longer shown anywhere


def test_the_reused_explanation_note_sits_in_the_collapsed_section_and_the_classifier_line_is_replaced():
    assert "splitExplanationNote(report.explanation).body" in APP and "splitExplanationNote(m.text).body" in APP
    block = APP[APP.index("{explanationNote && ("):]
    block = block[:block.index("</details>")]
    assert '<details className="how-decided">' in block and "<summary>How this was decided</summary>" in block
    assert "open" not in block.split(">")[0] and "{explanationNote}" in block
    assert "{prediction.labelled && <span>Rule check:</span>} {prediction.text}" in APP
    assert "No warning patterns detected in the recruiter's messages." in CHAT_VIEW


def test_the_note_text_matches_the_one_the_server_appends():
    assert chat_routes.REUSED_NOTE in CHAT_VIEW.replace('"\n  "', "")


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_node_tests_for_the_chat_view_pass():
    result = subprocess.run(["node", "--test", "src/services/chatView.test.mjs"], cwd=FRONTEND,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
