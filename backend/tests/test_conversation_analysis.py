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
