"""Display wording: 'Rule check' label, word-boundary quotes, plain guidance tags. No score, rule, API field or model change."""
import io
import shutil
import subprocess
from pathlib import Path

import pytest

flask = pytest.importorskip("flask")
from flask import Flask  # noqa: E402

from backend.routes import chat_routes  # noqa: E402
from backend.services.conversation_analysis import analyze_messages  # noqa: E402
from backend.services.conversation_parser import parse_chat_text  # noqa: E402
from backend.services.evidence import MESSAGE_EXCERPT_CHARS, build_evidence  # noqa: E402
from backend.services.text_display import ELLIPSIS, shorten_quote  # noqa: E402

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
APP = (FRONTEND / "src" / "App.jsx").read_text(encoding="utf-8")


def _whole_words_only(short: str, original: str) -> bool:
    """The shortened quote (without the ellipsis) ends at the end of a word of the original."""
    body = short[:-1] if short.endswith(ELLIPSIS) else short
    normal = " ".join(original.split())
    return normal.startswith(body) and (len(body) == len(normal) or not normal[len(body)].isalnum())


# --- item 7: quotes -------------------------------------------------------------------------------------------------

def test_short_text_is_unchanged_and_long_text_ends_at_a_word_with_an_ellipsis():
    assert shorten_quote("Pay the fee today.", 120) == "Pay the fee today."
    text = "Pay the registration fee of Rs 3000 within two hours or the seat goes to someone else, so please hurry now"
    short = shorten_quote(text, 60)
    assert short.endswith(ELLIPSIS) and len(short) <= 60 and _whole_words_only(short, text)
    assert short == "Pay the registration fee of Rs 3000 within two hours or…"


@pytest.mark.parametrize("limit", [10, 25, 40, 61, 119, 120, 159, 160])
def test_never_ends_mid_word_at_any_limit(limit):
    text = ("Recruiter says you must deposit the cheque, keep your first payment and transfer the remaining "
            "balance to our approved equipment supplier before Friday afternoon at the latest, without fail.")
    short = shorten_quote(text, limit)
    assert _whole_words_only(short, text), (limit, short)
    assert not short[:-1].rstrip().endswith((",", ";", ":", "-")) or not short.endswith(ELLIPSIS)


def test_punctuation_before_the_cut_is_dropped_and_whitespace_is_collapsed():
    assert shorten_quote("Send money,   then wait for the courier to arrive today", 18) == "Send money, then…"
    assert shorten_quote("one two, three four five six", 15) == "one two…"


def test_a_single_long_word_is_kept_whole_and_only_a_very_long_unbroken_run_is_cut():
    assert shorten_quote("Supercalifragilisticexpialidocious and more", 20) == "Supercalifragilisticexpialidocious…"
    blob = "x" * 500
    assert shorten_quote(blob, 50) == "x" * 49 + ELLIPSIS


def test_chat_rule_sign_quotes_and_message_excerpts_end_at_a_word():
    long_message = ("Pay the registration fee of Rs 3000 within 2 hours or the seat goes to someone else and you lose "
                    "the job offer completely, then send your bank details and add me on Telegram for onboarding today.")
    raw = f"24/10/26, 9:00 - Recruiter: {long_message}\n24/10/26, 9:01 - You: Why?\n"
    app = Flask(__name__)
    app.register_blueprint(chat_routes.chat_bp)
    body = app.test_client().post("/api/analyze-chat", data={"file": (io.BytesIO(raw.encode()), "chat.txt")},
                                  content_type="multipart/form-data").get_json()
    quotes = [r["matched_text"] for r in body["evidence"]["rule_signs"]]
    assert quotes and all(q.endswith(ELLIPSIS) and len(q) <= 120 and _whole_words_only(q, long_message) for q in quotes)
    message = body["evidence"]["messages"][0]["text"]
    assert len(long_message) > MESSAGE_EXCERPT_CHARS and message.endswith(ELLIPSIS) and _whole_words_only(message, long_message)


def test_flagged_message_excerpts_in_the_evidence_step_end_at_a_word():
    text = "word " * 60
    evidence = build_evidence("x", {}, [], flagged_messages=[{"time": "t", "speaker": "R", "sign": "s", "text": text}])
    excerpt = evidence["messages"][0]["text"]
    assert excerpt.endswith(ELLIPSIS) and excerpt[:-1].split()[-1] == "word" and len(excerpt) <= MESSAGE_EXCERPT_CHARS


def test_chat_scores_do_not_change():
    raw = ("24/10/26, 9:00 - Recruiter: Pay the registration fee of Rs 3000 within 2 hours. Send your bank details.\n"
           "24/10/26, 9:01 - You: Why?\n")
    result = analyze_messages(parse_chat_text(raw))
    assert (result["risk_score"], result["display_risk_score"], result["risk_category"]) == (80, 80, "Higher risk")


# --- item 6: label ------------------------------------------------------------------------------------------------

def test_the_chat_report_says_rule_check_not_classifier():
    assert "{prediction.labelled && <span>Rule check:</span>} {prediction.text}" in APP
    assert "<span>Classifier:</span>" not in APP


# --- item 8: guidance tags -----------------------------------------------------------------------------------------

@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_node_tests_for_the_plain_tags_pass():
    result = subprocess.run(["node", "--test", "src/services/textDisplay.test.mjs"], cwd=FRONTEND, capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
