"""Rule-based first-pass conversation warning detector; not a fraud verdict."""
from __future__ import annotations

import re
from .conversation_parser import ParsedMessage


def is_payment_request(text: str) -> bool:
    """Detect requests or instructions to make a payment, not questions about payment."""
    text = text.strip().lower()

    # Questions, refusals, and warnings about payment are not payment requests.
    non_request_patterns = [
        r"^\s*why\s+(?:do|should|must|would|am)\s+i\s+(?:need\s+to\s+)?pay\b",
        r"^\s*(?:do|should|must|can|could|would)\s+i\s+(?:need\s+to\s+)?pay\b",
        r"\bwhy\s+(?:are|am)\s+(?:you|i)\s+(?:asking|being asked)\s+to\s+pay\b",
        r"\b(?:i\s+)?(?:won't|will not|cannot|can't|refuse to)\s+pay\b",
        r"\b(?:send|share|forward|upload)\s+(?:me\s+)?(?:the\s+)?payment\s+(?:screenshot|proof|receipt)\b",
    ]

    if any(re.search(pattern, text) for pattern in non_request_patterns):
        return False

    # Look for an actual payment instruction or demand.
    payment_request_patterns = [
        r"\bpay\s+(?:(?:rs\.?|₹|\$)\s*)?[\d,]+",
        r"\bpay\s+(?:in\s+the\s+next\s+)?\d+\s*(?:minutes?|hours?|days?)\b",
        r"\bpay\s+(?:the\s+)?(?:fee|amount|charges)\b",
        r"\b(?:send|transfer|deposit)\s+(?:the\s+)?(?:money|payment|fee|amount)\b",
        r"\bregistration\s+fee\b",
        r"\bprocessing\s+fee\b",
        r"\bpayment\s+(?:is\s+)?required\b",
    ]

    return any(re.search(pattern, text) for pattern in payment_request_patterns)


# Payment requests must be explicit. A currency amount alone is not a payment
# demand: job postings commonly state salaries such as "Rs 45,000 per week".
PAYMENT_REQUEST_RE = re.compile(
    r"\b(?:pay|payment|fee|deposit|registration charge|processing charge|"
    r"security deposit|transfer|upi)\b",
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
                if label == "Payment request":
                    is_match = is_payment_request(text)
                else:
                    is_match = bool(pattern.search(text))

                if is_match:
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
            "Conversation risk scoring currently uses predefined rules; the DistilBERT classifier is used for job-posting analysis, not conversation classification.",
            "No external employer verification or reference-record retrieval was performed.",
        ],

    }
