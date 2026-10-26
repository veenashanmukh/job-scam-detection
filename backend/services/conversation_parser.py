"""Parse common WhatsApp and Telegram plain-text exports without storing them.

Supported input is a UTF-8 .txt export. The parser preserves multiline messages,
labels timestamped system events, and deliberately avoids guessing when a line
cannot be associated with a timestamped message.
"""
from __future__ import annotations

import re
from typing import TypedDict


class ParsedMessage(TypedDict):
    time: str
    speaker: str
    text: str
    is_system: bool

# Examples: 24/10/26, 14:30 - Name: hi | [24/10/26, 2:30 PM] Name: hi
# Also accepts YYYY-MM-DD timestamp exports and US month/day dates.
_PREFIXES = (
    re.compile(r"^\[?(?P<date>\d{1,4}[/-]\d{1,2}[/-]\d{2,4}),\s*(?P<clock>\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)\]?\s*(?:-|–)\s*(?P<body>.*)$"),
    re.compile(r"^\[?(?P<date>\d{1,4}[/-]\d{1,2}[/-]\d{2,4}),\s*(?P<clock>\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)\]?\s*(?P<body>.*)$"),
    re.compile(r"^\[?(?P<date>\d{1,2}\s+[A-Za-z]+\s+\d{4}),\s*(?P<clock>\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)\]?\s*(?:-|–|,)?\s*(?P<body>.*)$"),
    re.compile(r"^(?P<date>\d{1,2}\s+[A-Za-z]+\s+\d{4})\s+at\s+(?P<clock>\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?),?\s*(?P<body>.*)$", re.I),
    re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})[ T](?P<clock>\d{1,2}:\d{2}(?::\d{2})?)(?:\s*[-,]\s*|\s+)(?P<body>.*)$"),
)


def _parse_start(line: str):
    for pattern in _PREFIXES:
        match = pattern.match(line.strip())
        if match:
            return match.groupdict()
    return None


_NON_CHAT_FOOTER = re.compile(
    r"^this is a fictional sample conversation(?:\s|$)", re.I
)


def parse_chat_text(raw_text: str) -> list[ParsedMessage]:
    """Return timestamped messages; continuation lines belong to the prior message.

    Timestamped lines without a ``speaker: message`` separator are retained as
    system events. Untimestamped preamble lines are ignored rather than guessed.
    """
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string")
    messages: list[ParsedMessage] = []
    for line in raw_text.replace("\ufeff", "").splitlines():
        # Demo/sample disclaimers are file-level notes, not part of the last
        # timestamped message. Stop before attaching the disclaimer as a
        # continuation line to the conversation.
        if _NON_CHAT_FOOTER.match(line.strip()):
            break
        start = _parse_start(line)
        if start:
            body = (start.get("body") or "").strip()
            # WhatsApp/Telegram message text generally starts after 'Speaker:'.
            speaker, sep, message_text = body.partition(": ")
            if sep and speaker.strip() and len(speaker) <= 100:
                speaker = speaker.strip()
                text = message_text.strip()
                is_system = False
            else:
                speaker, text, is_system = "System", body, True
            messages.append({
                "time": f"{start['date']} {start['clock']}",
                "speaker": speaker,
                "text": text,
                "is_system": is_system,
            })
        elif messages and line.strip():
            # Preserve line breaks in message bodies; never attach un-timestamped
            # lines to a system event because that can merge unrelated notices.
            if not messages[-1]["is_system"]:
                messages[-1]["text"] += "\n" + line.strip()
    return messages
