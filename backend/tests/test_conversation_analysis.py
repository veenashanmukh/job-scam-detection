from backend.services.conversation_parser import parse_chat_text
from backend.services.conversation_analysis import analyze_messages, redact_personal_data


def test_detects_payment_sensitive_info_and_pressure():
    msgs = parse_chat_text(
        "24/10/26, 14:30 - Recruiter: Pay Rs 2,500 registration fee today.\n"
        "24/10/26, 14:31 - Recruiter: Send your Aadhaar and bank details within 10 minutes.\n"
    )
    result = analyze_messages(msgs)
    assert "Payment request" in result["warning_signs"]
    assert "Sensitive information request" in result["warning_signs"]
    assert "Pressure tactic" in result["warning_signs"]
    assert result["risk_score"] >= 60
    assert result["timeline"][1]["cumulative"] >= result["timeline"][0]["cumulative"]


def test_redacts_email_and_phone():
    text = redact_personal_data("Contact abc@example.com or call +91 98765 43210")
    assert "abc@example.com" not in text
    assert "98765 43210" not in text


def test_system_messages_do_not_add_risk():
    msgs = parse_chat_text("24/10/26, 14:30 - Messages and calls are end-to-end encrypted.")
    result = analyze_messages(msgs)
    assert result["risk_score"] == 0
    assert result["warning_signs"] == []



def test_pressure_tactics_written_in_words_and_common_phrases():
    msgs = parse_chat_text(
        "10/24/26, 10:02 AM - Rahul: Pay within two hours or give the job to someone else.\n"
        "10/24/26, 10:03 AM - Rahul: Offer expires soon. Do not miss your chance.\n"
    )
    result = analyze_messages(msgs)
    assert "Pressure tactic" in result["warning_signs"]
    assert all("Pressure tactic" in (item["sign"] or "") for item in result["timeline"])


def test_salary_amount_alone_is_not_a_payment_request():
    msgs = parse_chat_text(
        "10/24/26, 10:02 AM - Recruiter: Salary is Rs 45,000 per week.\n"
    )
    result = analyze_messages(msgs)
    assert "Payment request" not in result["warning_signs"]
    assert "Payment request" not in (result["timeline"][0]["sign"] or "")


def test_explicit_fee_demand_is_still_a_payment_request():
    msgs = parse_chat_text(
        "10/24/26, 10:02 AM - Recruiter: Pay Rs 2,500 registration fee today.\n"
    )
    result = analyze_messages(msgs)
    assert "Payment request" in result["warning_signs"]


def test_payment_screenshot_request_is_not_a_new_payment_demand():
    msgs = parse_chat_text(
        "10/24/26, 10:05 AM - Rahul: Send the payment screenshot to me on Telegram.\n"
    )
    result = analyze_messages(msgs)
    assert "Payment request" not in result["warning_signs"]
    assert "Moves to another app" in result["warning_signs"]


def test_sample_chat_footer_is_not_analyzed_as_a_message():
    raw = (
        "10/24/26, 10:09 AM - Rahul: Do not miss your chance.\n\n"
        "This is a fictional sample conversation created for software testing. "
        "Do not send money or personal information.\n"
    )
    msgs = parse_chat_text(raw)
    assert len(msgs) == 1
    assert "fictional sample conversation" not in msgs[0]["text"].lower()
    result = analyze_messages(msgs)
    assert "Pressure tactic" in result["warning_signs"]
    assert "Payment request" not in result["warning_signs"]


# --- pressure phrasings (item 2) -------------------------------------------------

import pytest  # noqa: E402

from backend.services.conversation_analysis import (  # noqa: E402
    detect_self, is_pressure, proposes_moving_app,
)


@pytest.mark.parametrize("text", [
    "Please send your bank account number and ID card today, it is urgent.",
    "Reply ASAP or the offer goes to someone else.",
    "You must pay within 2 hours to keep your seat.",
    "Hurry, only 3 seats left!",
    "Transfer the fee today to confirm your slot.",
    "Last chance to join, the deadline is tonight.",
])
def test_pressure_phrasings_are_detected(text):
    assert is_pressure(text)


@pytest.mark.parametrize("text", [
    "Can we talk today about the role?",
    "There is no urgency, take your time.",
    "It is not urgent, reply whenever you can.",
    "The interview is scheduled for next Monday.",
])
def test_non_pressure_sentences_are_not_flagged(text):
    assert not is_pressure(text)


def test_reported_sentence_now_triggers_pressure_in_a_chat():
    msgs = parse_chat_text("24/10/26, 14:30 - Recruiter: Please send your bank account number and ID card today, it is urgent.\n")
    result = analyze_messages(msgs)
    assert {"Pressure tactic", "Sensitive information request"} <= set(result["warning_signs"])


# --- only the other party is scored (item 3) -----------------------------------------

CHAT = (
    "24/10/26, 9:00 - Recruiter: Hello, we have a data entry role for you.\n"
    "24/10/26, 9:01 - Priya: Do I need to pay anything? Why should I share my bank details?\n"
    "24/10/26, 9:02 - Priya: I will not pay a registration fee today, it is urgent for me to know.\n"
    "24/10/26, 9:03 - Recruiter: Salary is good. Can you start Monday?\n"
)


def test_user_is_the_speaker_who_is_not_the_recruiter_and_their_messages_are_not_scored():
    result = analyze_messages(parse_chat_text(CHAT))
    assert result["you"] == "Priya" and result["you_source"] == "not_recruiter"
    assert result["participants"] == ["Recruiter", "Priya"]
    assert result["risk_score"] == 0 and result["warning_signs"] == []
    assert [t["own"] for t in result["timeline"]] == [False, True, True, False]
    assert any("Priya were treated as your own" in n for n in result["uncertainty_notes"])


def test_without_exclusion_the_same_messages_would_score():
    msgs = parse_chat_text(CHAT)
    scored_all = analyze_messages(msgs, self_name="Nobody")  # unknown name: nobody excluded
    assert scored_all["you"] is None and scored_all["you_source"] == "selected_not_found"
    assert scored_all["risk_score"] > 0


def test_user_can_choose_their_name_case_insensitively():
    result = analyze_messages(parse_chat_text(CHAT), self_name=" recruiter ")
    assert result["you"] == "Recruiter" and result["you_source"] == "selected"
    assert [t["own"] for t in result["timeline"]] == [True, False, False, True]


@pytest.mark.parametrize("raw, you, source", [
    ("24/10/26, 9:00 - You: Hi\n24/10/26, 9:01 - HR: Pay the fee now\n", "You", "export_label"),
    ("24/10/26, 9:00 - HR: Pay the fee now\n", None, "single_speaker"),
    ("24/10/26, 9:00 - A: Hello\n24/10/26, 9:01 - B: Hi there\n", None, "undetermined"),
    ("24/10/26, 9:00 - A: Can I ask something?\n24/10/26, 9:01 - B: May I know more?\n", None, "undetermined"),
])
def test_detect_self_edge_cases(raw, you, source):
    assert detect_self(parse_chat_text(raw)) == (you, source)


def test_undetermined_user_scores_everyone_and_says_so():
    result = analyze_messages(parse_chat_text("24/10/26, 9:00 - A: Hello\n24/10/26, 9:01 - B: Pay the registration fee now\n"))
    assert result["you"] is None and "Payment request" in result["warning_signs"]
    assert any("Choose your name" in n for n in result["uncertainty_notes"])


# --- moving to another app only when proposed (item 4) --------------------------------

@pytest.mark.parametrize("text", [
    "Let's continue on Telegram.",
    "Add me on WhatsApp for the next steps.",
    "Message me on Telegram, my handle is in the profile.",
    "Move to Telegram for onboarding.",
    "Please join our Telegram group.",
    "Send the payment screenshot to me on Telegram.",
])
def test_proposals_to_move_apps_are_detected(text):
    assert proposes_moving_app(text)


@pytest.mark.parametrize("text", [
    "I saw your advert on WhatsApp.",
    "Is this job also posted on Telegram?",
    "We never use WhatsApp for interviews.",
    "Our company has a Telegram channel for news.",
])
def test_mentions_of_apps_are_not_proposals(text):
    assert not proposes_moving_app(text)
    result = analyze_messages(parse_chat_text(f"24/10/26, 9:00 - HR: {text}\n"))
    assert "Moves to another app" not in result["warning_signs"]
