"""Rule-based first-pass conversation warning detector; not a fraud verdict."""
from __future__ import annotations

import re
from .conversation_parser import ParsedMessage

# Payment requests must be explicit. A currency amount alone is not a payment
# demand: job postings commonly state salaries such as "Rs 45,000 per week".
PAYMENT_REQUEST_RE = re.compile(
    r"\b(?:pay|payment(?!\s+(?:screenshot|proof|receipt))|fee|deposit|registration charge|processing charge|"
    r"security deposit|transfer|upi|send\s+(?:me\s+)?(?:rs\.?|₹|\$|money))\b",
    re.I,
)

RULES: list[tuple[str, re.Pattern[str], int]] = [
    ("Payment request", PAYMENT_REQUEST_RE, 35),
    ("Sensitive information request", re.compile(r"\b(?:aadhaar|aadhar|pan card|bank details|account number|card number|cvv|otp|password|passport|government id|identity proof|send your id|upload your id)\b", re.I), 25),
    ("Pressure tactic", re.compile(
        r"\b(?:within\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\w+-\w+)\s*(?:minutes?|hours?)|"
        r"immediately|right now|last chance|limited seats?|limited slots?|act fast|today only|"
        r"or you will lose|seat goes to someone|give the job to someone else|"
        r"offer expires soon|offer expires in|expires within|do not miss your chance|"
        r"don't miss your chance|don't delay|do not delay|if you delay)\b", re.I
    ), 20),
    ("Moves to another app", re.compile(r"\b(?:telegram|whatsapp|signal|私聊|move to (?:another|our) app|continue on)\b", re.I), 10),
    ("Unrealistic job promise", re.compile(r"\b(?:guaranteed (?:job|placement|income)|no interview|without interview|earn\s+(?:₹|rs\.?\s*)?\d[\d,]*(?:\s*(?:per|/))\s*(?:day|week|month)|work\s+from\s+home.{0,50}(?:easy money|guaranteed))\b", re.I), 15),
]

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)")


def redact_personal_data(text: str) -> str:
    """Mask email addresses and phone-like sequences before API response display."""
    text = EMAIL_RE.sub("[REDACTED EMAIL]", text)
    return PHONE_RE.sub("[REDACTED PHONE]", text)


def analyze_messages(messages: list[ParsedMessage]) -> dict:
    """Analyze each message and build a monotonically increasing warning timeline."""
    timeline = []
    all_signs: list[str] = []
    score = 0
    for message in messages:
        text = message["text"]
        matched = []
        weight = 0
        if not message.get("is_system", False):
            for label, pattern, points in RULES:
                if pattern.search(text):
                    matched.append(label)
                    weight += points
                    if label not in all_signs:
                        all_signs.append(label)
        # Cap at 100; the score is a heuristic, not a probability.
        score = min(100, score + weight)
        timeline.append({
            "time": message["time"],
            "speaker": message["speaker"],
            "text": redact_personal_data(text),
            "sign": "; ".join(matched) if matched else None,
            "cumulative": score,
        })
    category = "Lower risk" if score < 30 else "Moderate risk" if score < 60 else "Higher risk"
    return {
        "risk_score": score,
        "risk_category": category,
        "classifier_prediction": "Potentially fraudulent conversation" if score >= 60 else "Warning signs require review" if score >= 30 else "No strong rule-based warning pattern detected",
        "warning_signs": all_signs,
        "supporting_evidence": [],
        "timeline": timeline,
        "explanation": (
            "The report highlights message patterns that may warrant caution. "
            "The score is a rule-based indicator, not a probability or proof of fraud."
        ),
        "uncertainty_notes": [
            "This first version uses rules only; it is not a trained conversation classifier.",
            "No external employer verification or reference-record retrieval was performed.",
        ],
    }
