from backend.services.conversation_parser import parse_chat_text


def test_whatsapp_export_and_multiline_message():
    raw = (
        "24/10/26, 14:30 - Recruiter: Hello\n"
        "Please send your CV\n"
        "24/10/26, 14:31 - You: Sure\n"
    )
    messages = parse_chat_text(raw)
    assert len(messages) == 2
    assert messages[0]["speaker"] == "Recruiter"
    assert messages[0]["text"] == "Hello\nPlease send your CV"
    assert messages[1]["speaker"] == "You"
    assert not messages[0]["is_system"]


def test_bracketed_timestamp_and_system_line():
    raw = "[24/10/26, 2:30 PM] Recruiter: Hello\n[24/10/26, 2:31 PM] Messages and calls are end-to-end encrypted.\n"
    messages = parse_chat_text(raw)
    assert len(messages) == 2
    assert messages[0]["speaker"] == "Recruiter"
    assert messages[1]["is_system"] is True


def test_iso_timestamp_and_empty_input():
    assert parse_chat_text("2026-10-24 14:30:00 Recruiter: Hi")[0]["text"] == "Hi"
    assert parse_chat_text("unrecognized export") == []


def test_telegram_desktop_style_timestamp():
    messages = parse_chat_text("24 October 2026 at 14:30, Recruiter: Pay a registration fee")
    assert len(messages) == 1
    assert messages[0]["speaker"] == "Recruiter"
    assert messages[0]["text"] == "Pay a registration fee"
