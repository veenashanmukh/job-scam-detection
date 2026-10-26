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
