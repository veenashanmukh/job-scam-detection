"""Rule-based first-pass conversation warning detector; not a fraud verdict."""
from __future__ import annotations

import re

from .conversation_parser import ParsedMessage
from .score_display import clamp_display_score


# A sentence that says the company will NOT ask for money / data is a reassurance or a warning.
NEGATION_GUARD_RE = re.compile(
    r"\b(?:never\s+(?:ask|asks|asked|charge|charges|request|requests|share|give|send|tell)|"
    r"(?:do|does|did|should|must|will|would|can|could)\s*(?:not|n't)\s+(?:ask|pay|charge|share|give|send|tell|provide|need)|"
    r"(?:don't|doesn't|won't|wouldn't|shouldn't|mustn't|can't|cannot)\s+(?:ask|pay|charge|share|give|send|tell|provide|need)|"
    r"no\s+(?:\w+\s+){0,2}(?:fees?|charges?|payments?|deposits?|costs?)|"
    r"(?:not|isn't|is not|aren't|are not)\s+(?:required|needed|necessary|asked)|"
    r"we\s+(?:will\s+)?not\b|free\s+of\s+charge|without\s+(?:any\s+)?(?:fee|payment|charge)s?|"
    r"nothing\s+to\s+pay|no\s+need\s+to\s+(?:pay|share|send|give))",
    re.I,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|[\n;]+")
# Fee-like nouns and money-moving verbs for the broadened payment rule.
_FEE_NOUN = (r"(?:fees?|charges?|deposits?|costs?|contributions?|courier|insurance|documentation|"
             r"processing|activation|training|registration|kit|dues)")
_PAY_VERB = r"(?:pay|paying|send|sending|transfer|transferring|deposit|depositing|remit|remitting|wire|wiring)"
# "we pay ..." / "the company will send ..." describe what the company does, not a demand on the user.
_COMPANY_SUBJECT_RE = re.compile(r"\b(?:we|i|they|company|employer|firm|agency)(?:'ll|'d)?\s+(?:will\s+|shall\s+|do\s+|can\s+|may\s+)?$", re.I)
_BROAD_PAYMENT_RE = [
    re.compile(rf"\b{_PAY_VERB}\b[^.!?\n]{{0,60}}?\b{_FEE_NOUN}\b", re.I),
    re.compile(rf"\b{_FEE_NOUN}\b[^.!?\n]{{0,40}}?\b(?:must|has to|needs to|should|to)\s+be\s+(?:paid|sent|transferred|deposited|remitted)\b", re.I),
    re.compile(r"\b(?:buy|purchase|get|send|share|redeem)\b[^.!?\n]{0,40}?\b(?:gift\s*cards?|vouchers?|itunes\s+cards?|google\s+play\s+cards?|steam\s+cards?)\b", re.I),
    re.compile(r"\b(?:buy|purchase|pay\s+for|get)\b[^.!?\n]{0,60}?\b(?:and|then)\b[^.!?\n]{0,25}?\b(?:we|i|company)\b[^.!?\n]{0,15}?\b(?:refund|reimburse|pay\s+you\s+back|repay)\b", re.I),
    re.compile(r"\b(?:return|send|refund|wire|transfer|remit)\b[^.!?\n]{0,40}?\b(?:overpaid|overpayment|(?:extra|excess|surplus)\s+(?:money|amount|funds|balance|cash|payment))\b|\b(?:extra|excess|surplus)\s+(?:money|amount|funds|balance|cash)\b[^.!?\n]{0,40}?\b(?:return|send\s+back|refund|transfer\s+back)\b", re.I),
]


COMPANY_SUBJECT_RE = _COMPANY_SUBJECT_RE  # also used by the job-posting rules


def _sentences(text: str) -> list[str]:
    return [part for part in _SENTENCE_SPLIT_RE.split(text) if part and part.strip()]


def _broad_payment_match(sentence: str) -> bool:
    for match in _BROAD_PAYMENT_RE[0].finditer(sentence):
        if not _COMPANY_SUBJECT_RE.search(sentence[max(0, match.start() - 20):match.start()]):
            return True
    return any(p.search(sentence) for p in _BROAD_PAYMENT_RE[1:])


def is_payment_request(text: str) -> bool:
    """True if any sentence demands a payment; sentences that deny asking for money do not count."""
    return any(_sentence_is_payment_request(sentence) for sentence in _sentences(text))


def _sentence_is_payment_request(text: str) -> bool:
    """Detect requests or instructions to make a payment, not questions about payment."""
    text = text.strip().lower()
    if NEGATION_GUARD_RE.search(text):
        return False

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

    return any(re.search(pattern, text) for pattern in payment_request_patterns) or _broad_payment_match(text)


# Payment requests must be explicit. A currency amount alone is not a payment
# demand: job postings commonly state salaries such as "Rs 45,000 per week".
PAYMENT_REQUEST_RE = re.compile(
    r"\b(?:pay|payment|fee|deposit|registration charge|processing charge|"
    r"security deposit|transfer|upi)\b",
    re.I,
)

PRESSURE_RE = re.compile(
    r"\b(?:within\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\w+-\w+)\s*(?:minutes?|hours?)|"
    r"immediately|right now|last chance|limited seats?|limited slots?|act fast|today only|"
    r"or you will lose|seat goes to someone|give the job to someone else|goes to someone else|"
    r"offer expires soon|offer expires in|expires within|do not miss your chance|"
    r"don't miss your chance|don't delay|do not delay|if you delay|"
    r"urgent(?:ly)?|asap|as soon as possible|hurry|"
    r"only\s+\d+\s+(?:seats?|slots?|spots?|places?|positions?)\s+left|"
    r"(?:deadline|offer)\s+(?:is|ends|expires)\s+(?:today|tonight)|before\s+(?:the\s+)?end\s+of\s+(?:the\s+)?day|"
    r"(?:send|pay|transfer|deposit|submit)\b[^.!?]{0,60}\b(?:today|tonight|now))\b",
    re.I,
)
# A pressure phrase that is negated ("it is not urgent", "no need to hurry") is not pressure.
_PRESSURE_NEGATED = re.compile(r"\b(?:not|no|never|isn't|no need to|don't need to)\s+(?:\w+\s+){0,2}$", re.I)

APPS = r"(?:telegram|whatsapp|signal|wechat|skype|viber|line app|another app|our app|this number)"
# Only a proposal to move the conversation counts, not any mention of an app.
MOVE_APP_RE = re.compile(
    rf"\b(?:let'?s|let us|we can|we will|we'll|please|kindly|you can|you should)?\s*"
    rf"(?:continue|move|chat|talk|switch|connect|communicate)\s+(?:this\s+|the\s+chat\s+)?(?:on|to|over to|via|in)\s+{APPS}"
    rf"|\b(?:add|message|text|contact|reach|dm|ping|call|join|find|follow)\s+(?:me|us|our\s+\w+|my\s+\w+)?\s*(?:on|via|at|in)\s+{APPS}"
    rf"|\bjoin\s+(?:our|my|the)\s+{APPS}\s+(?:group|channel)"
    rf"|\b(?:send|share|forward)\b[^.!?]{{0,60}}\b(?:on|via|to me on)\s+{APPS}"
    rf"|私聊",
    re.I,
)


def is_pressure(text: str) -> bool:
    """Urgency or deadline pressure, ignoring negated phrases such as 'it is not urgent'."""
    for match in PRESSURE_RE.finditer(text):
        if not _PRESSURE_NEGATED.search(text[max(0, match.start() - 30):match.start()]):
            return True
    return False


def proposes_moving_app(text: str) -> bool:
    return bool(MOVE_APP_RE.search(text))


SENSITIVE_INFO_RE = re.compile(
    r"\b(?:aadhaar|aadhar|pan card|bank details|account number|card number|cvv|otp|password|passport|government id|identity proof|send your id|upload your id|"
    r"bank\s+(?:login|log-in|credentials?)|(?:net|online|internet|mobile)\s*banking\s+(?:login|credentials?|password|details|username)|login\s+(?:details|credentials?)|credentials?|"
    r"(?:debit|credit|atm)\s+card|card\s+(?:expiry|expiration|number|details)|expiry\s+date|cvc|atm\s+pin|card\s+pin|one[- ]time\s+(?:password|code|pin)|"
    r"sms\s+code|verification\s+code|security\s+code|"
    r"social\s+security|ssn|national\s+insurance\s+number|"
    r"(?:passport|id|identity|driving\s+licen[cs]e|driver'?s\s+licen[cs]e)\s+(?:photo|picture|copy|scan|image)s?|(?:photo|picture|copy|scan|image)s?\s+of\s+your\s+(?:passport|id|identity\s+card|driving\s+licen[cs]e)|selfie|"
    r"mother'?s\s+maiden\s+name)\b",
    re.I,
)


def asks_sensitive_information(text: str) -> bool:
    """Sensitive-data request, ignoring sentences that tell the reader never to share it."""
    return any(SENSITIVE_INFO_RE.search(s) and not NEGATION_GUARD_RE.search(s) for s in _sentences(text))


RULES: list[tuple[str, re.Pattern[str], int]] = [
    ("Payment request", PAYMENT_REQUEST_RE, 35),
    ("Sensitive information request", SENSITIVE_INFO_RE, 25),
    ("Pressure tactic", PRESSURE_RE, 20),
    ("Moves to another app", MOVE_APP_RE, 10),
    ("Unrealistic job promise", re.compile(r"\b(?:guaranteed (?:job|placement|income)|no interview|without interview|earn\s+(?:₹|rs\.?\s*)?\d[\d,]*(?:\s*(?:per|/))\s*(?:day|week|month)|work\s+from\s+home.{0,50}(?:easy money|guaranteed))\b", re.I), 15),
]

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)")


def redact_personal_data(text: str) -> str:
    """Mask email addresses and phone-like sequences before API response display."""
    text = EMAIL_RE.sub("[REDACTED EMAIL]", text)
    return PHONE_RE.sub("[REDACTED PHONE]", text)


# Labels of the account owner in an export or a pasted chat (exact label), and labels of the recruiter side (any word).
SELF_LABELS = {"you", "me", "myself", "candidate", "applicant", "seeker"}
RECRUITER_WORDS = {"recruiter", "hr", "agent", "employer", "manager"}  # also "Hiring Manager" via "manager"


def participants(messages: list[ParsedMessage]) -> list[str]:
    return list(dict.fromkeys(m["speaker"] for m in messages if not m.get("is_system", False)))


def _is_recruiter_side(label: str) -> bool:
    return any(word in RECRUITER_WORDS for word in re.findall(r"[a-z]+", label.casefold()))


def detect_self(messages: list[ParsedMessage], self_name: str | None = None) -> tuple[str | None, str]:
    """
    Return (speaker treated as the user, how it was chosen).

    A chosen name wins if it matches a participant. Otherwise exactly one speaker labelled You, Me,
    Myself, Candidate, Applicant or Seeker; otherwise, when exactly one speaker is NOT on the recruiter
    side (Recruiter, HR, Hiring Manager, Agent, Employer, Manager), that speaker. Anything still
    ambiguous returns nobody (so every message is scored and the user is asked to pick); it never guesses.
    """
    people = participants(messages)
    if self_name is not None and str(self_name).strip():
        wanted = str(self_name).strip().casefold()
        match = next((p for p in people if p.strip().casefold() == wanted), None)
        return (match, "selected") if match else (None, "selected_not_found")
    if len(people) < 2:
        return None, "single_speaker"
    labelled = [p for p in people if p.strip().casefold() in SELF_LABELS]
    if len(labelled) == 1:
        return labelled[0], "export_label"
    if not labelled:
        others = [p for p in people if not _is_recruiter_side(p)]
        if len(others) == 1 and len(others) < len(people):
            return others[0], "not_recruiter"
    return None, "undetermined"


KEYWORD_LIMITATION_NOTE = (
    "This score uses keyword rules only. It can miss scams that do not use explicit payment, "
    "personal-data or pressure wording, so a low score is not a sign that a chat is safe."
)


def analyze_messages(messages: list[ParsedMessage], self_name: str | None = None) -> dict:
    """Analyze the other party's messages and build a monotonically increasing warning timeline."""
    you, you_source = detect_self(messages, self_name)
    timeline = []
    all_signs: list[str] = []
    score = 0
    for message in messages:
        text = message["text"]
        matched = []
        weight = 0
        own = you is not None and message["speaker"] == you
        if not message.get("is_system", False) and not own:
            for label, pattern, points in RULES:
                if label == "Payment request":
                    is_match = is_payment_request(text)
                elif label == "Sensitive information request":
                    is_match = asks_sensitive_information(text)
                elif label == "Pressure tactic":
                    is_match = is_pressure(text)
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
            "own": own,
        })
    category = "Lower risk" if score < 30 else "Moderate risk" if score < 60 else "Higher risk"
    if you is not None:
        who_note = f"Messages from {you} were treated as your own and were not scored."
    else:
        who_note = ("It was not clear which participant is you, so every message was scored. "
                    "Choose your name to exclude your own messages.")
    return {
        "participants": participants(messages),
        "you": you,
        "you_source": you_source,
        "risk_score": score,
        "display_risk_score": clamp_display_score(score),  # the number to show: never exactly 0 or 100
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
            "No external employer verification was performed.",
            KEYWORD_LIMITATION_NOTE,
            who_note,
        ],
    }
