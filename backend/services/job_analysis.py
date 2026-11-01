
"""Turn a DistilBERT prediction into a report with job-posting warning signs."""
from __future__ import annotations

import math
import re
from typing import NamedTuple

from backend.services.conversation_analysis import COMPANY_SUBJECT_RE, NEGATION_GUARD_RE
from backend.services.score_display import clamp_display_score

PROVISIONAL_SCORE_NOTE = (
    "The risk score comes from the classifier's log-odds, not a probability, and does not "
    "use the warning signs. It is not a calibrated probability of fraud."
)

MODEL_LIMITS_NOTE = (
    "The classifier can miss some scams and can flag some legitimate postings. "
    "It reads the text only and does not check the company."
)

WARNING_RULES = [
    (
        "Upfront payment",
        re.compile(
            r"\b(registration fee|application fee|upfront fee|pay (?:a |an )?"
            r"(?:fee|deposit)|payment before (?:starting|you start|employment))\b",
            re.IGNORECASE,
        ),
    ),
        (
        "Pressure tactic",
        re.compile(
            r"\b(pay\s+(?:a\s+)?(?:refundable\s+)?(?:registration\s+)?"
            r"(?:application\s+)?fee\s+today|"
            r"pay today|pay immediately|act now|urgent(?:ly)?|"
            r"limited vacancies|secure your position|offer expires|"
            r"immediate action required|lose the offer|"
            r"or lose (?:the )?(?:offer|position)|"
            r"today to secure your position)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Guaranteed employment",
        re.compile(
            r"\b(guaranteed employment|guaranteed (?:job|position|hiring)|"
            r"job guaranteed after payment)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "No interview",
        re.compile(
            r"\b(no interview (?:is )?(?:required|needed)|without an interview|"
            r"no interview process)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Unusually high salary claim",
        re.compile(
            r"\b(high salary|earn\s+\$?\s*\d[\d,]*(?:\s*[-–]\s*\$?\s*\d[\d,]*)?"
            r"\s*(?:per month|monthly)|"
            r"\$?\s*\d[\d,]*(?:\s*[-–]\s*\$?\s*\d[\d,]*)?"
            r"\s*(?:per month|monthly))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Requests personal information",
        re.compile(
            r"\b(send|share|provide)\b.{0,80}\b(personal details|"
            r"bank details|identity documents|social security number)\b",
            re.IGNORECASE,
        ),
    ),
]

class CategoryRule(NamedTuple):
    """A category-level rule added on top of WARNING_RULES. The label is shared with the base rule if one exists."""
    label: str
    pattern: re.Pattern
    company_subject: bool = False        # ignore "we pay ...", "the company will send ..."
    extra_guard: re.Pattern | None = None  # a sentence matching this is not a request (for example after an offer)
    negation_guard: bool = True          # ignore sentences that say the employer will NOT ask for this


_PAY = r"(?:pay|paying|send|sending|transfer|transferring|deposit|depositing|remit|remitting|wire|wiring)"
_FEE_KIND = (r"(?:training|activation|registration|processing|courier|insurance|visa|medical|documentation|"
             r"security|joining|application|equipment|starter|admin(?:istration)?|clearance)")
_SENSITIVE_ITEM = (r"(?:bank\s+(?:login|log-in|credentials?)|(?:net|online|internet|mobile)\s*banking\s+"
                   r"(?:login|credentials?|password|details|username)|card\s+(?:number|expiry|expiration|details)|"
                   r"(?:debit|credit|atm)\s+card|expiry\s+date|cvv|cvc|otp|one[- ]time\s+(?:password|code|pin)|"
                   r"sms\s+code|verification\s+code|security\s+code|passport|"
                   r"(?:id|identity|driving\s+licen[cs]e|driver'?s\s+licen[cs]e)\s+(?:card|photo|picture|copy|scan|image)s?|"
                   r"(?:photos?|pictures?|copy|copies|scans?|images?)\s+of\s+(?:your\s+)?(?:id|passport|identity\s+card|driving\s+licen[cs]e)|"
                   r"photo\s+id|selfie|social\s+security|ssn)")
_AFTER_OFFER_RE = re.compile(
    r"\b(?:after|once|when)\s+(?:you\s+)?(?:have\s+)?(?:accept(?:ed)?|receive[d]?|are\s+hired|joined|an\s+offer|your\s+offer|"
    r"hired|selected)|\bon\s+(?:your\s+)?(?:first\s+day|joining|day\s+one)|\bat\s+(?:onboarding|joining)|\bonboarding\b|"
    r"right[- ]to[- ]work|background\s+check|\bI-?9\b",
    re.IGNORECASE,
)

# Existing patterns and labels above are kept as they are; these add coverage by category.
CATEGORY_RULES = [
    # Cheque or overpayment, then moving the money on.
    CategoryRule("Cheque or overpayment scam", re.compile(
        r"(?:\bcheques?\b|\b(?:a|the|your|this|that|our|first|pay)\s+check\b|\bmoney\s+orders?\b|\bbank\s+drafts?\b)"
        r"[\s\S]{0,250}?\b(?:wire|transfer|forward|remit|send\s+on|pass\s+on|pay)\b[\s\S]{0,60}?"
        r"\b(?:balance|remainder|remaining|rest|excess|extra|difference|supplier|vendor|third[- ]party|another\s+account)\b",
        re.IGNORECASE)),
    CategoryRule("Cheque or overpayment scam", re.compile(
        r"\b(?:overpay(?:ment|ed)?|over-?paid|excess\s+(?:funds|amount|payment)|extra\s+(?:funds|amount))\b"
        r"[\s\S]{0,120}?\b(?:wire|transfer|send|return|forward|remit)\b", re.IGNORECASE)),
    # A fee, charge or deposit the applicant has to pay.
    CategoryRule("Upfront payment", re.compile(
        rf"\b{_PAY}\b[^.!?\n]{{0,60}}?\b(?:{_FEE_KIND}\s+(?:fees?|charges?|deposits?|costs?|payments?|kit)|"
        rf"(?:fees?|charges?|cost)|(?:security|refundable|advance|upfront|initial|small|one-time)\s+deposit)\b",
        re.IGNORECASE), company_subject=True),
    CategoryRule("Upfront payment", re.compile(
        rf"\b{_PAY}\s+(?:a\s+|an\s+)?(?:\w+\s+)?(?:rs\.?|₹|\$|£|€|usd|inr)?\s*\d[\d,]*(?:\.\d+)?\s+{_FEE_KIND}\b",
        re.IGNORECASE), company_subject=True),
    CategoryRule("Upfront payment", re.compile(
        r"\b(?:fees?|charges?|deposits?|costs?)\b[^.!?\n]{0,40}?\b(?:must|has\s+to|have\s+to|needs?\s+to|should)\s+be\s+"
        r"(?:paid|sent|transferred|deposited|remitted|settled)\b", re.IGNORECASE)),
    CategoryRule("Upfront payment", re.compile(
        rf"\b{_FEE_KIND}\s+(?:fees?|charges?)\s+(?:of|is|are|:)\s*(?:rs\.?|₹|\$|£|€|usd|inr)?\s*\d", re.IGNORECASE)),
    # Gift cards or crypto to unlock work.
    CategoryRule("Gift card or crypto payment", re.compile(
        r"\b(?:buy|purchase|send|share|redeem|load|top[- ]?up|scratch|read\s+out)\b[^.!?\n]{0,50}?"
        r"\b(?:gift\s*cards?|(?:itunes|google\s+play|steam|amazon)\s+(?:cards?|vouchers?|codes?)|prepaid\s+cards?|"
        r"voucher\s+codes?)\b", re.IGNORECASE), company_subject=True),
    CategoryRule("Gift card or crypto payment", re.compile(
        r"\b(?:deposit|top[- ]?up|fund|buy|transfer|send|pay)\b[^.!?\n]{0,50}?"
        r"\b(?:crypto(?:currency)?|bitcoin|btc|usdt|ethereum|tether)\b[^.!?\n]{0,80}?"
        r"\b(?:unlock|activate|start\s+(?:working|earning)|access\s+(?:your\s+)?(?:tasks?|account|work|orders?)|"
        r"before\s+(?:you\s+)?(?:can|start))\b", re.IGNORECASE)),
    CategoryRule("Gift card or crypto payment", re.compile(
        r"\b(?:starter|initial|opening|activation|task|security)\s+(?:balance|deposit|amount|funds?)\b[^.!?\n]{0,60}?"
        r"\b(?:usdt|crypto(?:currency)?|bitcoin|btc|ethereum|tether)\b", re.IGNORECASE)),
    CategoryRule("Gift card or crypto payment", re.compile(
        r"\b(?:unlock|activate)\b[^.!?\n]{0,40}?\b(?:tasks?|account|work|orders?)\b[^.!?\n]{0,60}?"
        r"\b(?:deposit|top[- ]?up|fund)\b[^.!?\n]{0,40}?\b(?:crypto(?:currency)?|bitcoin|btc|usdt|ethereum|tether)\b",
        re.IGNORECASE)),
    # More ways to say there is no interview. (No negation guard: "interview not required" is the signal itself.)
    CategoryRule("No interview", re.compile(
        r"\bno\s+(?:\w+\s+){0,3}(?:or|and|nor)\s+interviews?\s+(?:is\s+|are\s+|will\s+be\s+)?(?:required|needed|necessary)\b|"
        r"\bno\s+interviews?\s+(?:is\s+|are\s+|will\s+be\s+)?(?:required|needed|necessary|involved|conducted)\b|"
        r"\bwithout\s+(?:any\s+|an\s+|the\s+)?interviews?\b|"
        r"\binterviews?\s+(?:is\s+|are\s+)?(?:not|n't)\s+(?:required|needed|necessary)\b|"
        r"\b(?:skip|bypass|waive)\s+(?:the\s+)?interviews?\b|"
        r"\b(?:do|does)\s*n[o']t\s+(?:need|require)\s+(?:an?\s+)?interview\b|"
        r"\bno\s+interviews?(?=\s*(?:[,.;!]|$)|\s+(?:and|or|no|just|only|at\s+all))|"
        r"\byou(?:'re|\s+are)\s+hired\s+(?:instantly|immediately|on\s+the\s+spot)\b", re.IGNORECASE), negation_guard=False),
    # Sensitive data asked for up front.
    CategoryRule("Requests personal information", re.compile(
        rf"\b(?:send|share|provide|upload|submit|email|text|give|tell|forward|reply\s+with|supply|need|needs|require|requires|"
        rf"want|wants|ask(?:ing)?\s+for|request(?:ing)?)\b[^.!?\n]{{0,80}}?"
        rf"\b{_SENSITIVE_ITEM}\b", re.IGNORECASE), extra_guard=_AFTER_OFFER_RE),
    # Reshipping parcels or passing on money for a share.
    CategoryRule("Parcel or money forwarding", re.compile(
        r"\b(?:re-?ship(?:ping)?|repack(?:age|aging)?)\b|"
        r"\b(?:receive|accept|get)\b[^.!?\n]{0,40}?\b(?:parcels?|packages?|packets?|shipments?)\b[^.!?\n]{0,50}?"
        r"\b(?:at|to)\s+your\s+(?:home|address|house|place)\b[^.!?\n]{0,60}?\b(?:forward|ship|send|post|mail)\b|"
        r"\bforward(?:ing)?\b[^.!?\n]{0,40}?\b(?:parcels?|packages?|packets?|shipments?)\b[^.!?\n]{0,40}?"
        r"\b(?:abroad|overseas|another\s+address|our\s+(?:client|customer|agent)s?)\b", re.IGNORECASE)),
    CategoryRule("Parcel or money forwarding", re.compile(
        r"\b(?:forward|pass\s+on|send\s+on|transfer\s+on|move\s+on)\b[^.!?\n]{0,50}?"
        r"\b(?:payments?|transfers?|funds|money|deposits?)\b[^.!?\n]{0,80}?"
        r"\b(?:keep|commission|your\s+(?:share|cut|fee)|percent|%)", re.IGNORECASE)),
    CategoryRule("Parcel or money forwarding", re.compile(
        r"\b(?:receive|accept|get)\b[^.!?\n]{0,40}?\b(?:payments?|transfers?|funds)\b[^.!?\n]{0,60}?"
        r"\bon\s+behalf\s+of\b[^.!?\n]{0,60}?\b(?:forward|send|transfer|pass)\b", re.IGNORECASE)),
]


# The chat guard plus "never be asked / required / charged to ..." wording common in postings.
_JOB_NEGATION_GUARD_RE = re.compile(
    NEGATION_GUARD_RE.pattern + r"|\bnever\s+(?:\w+\s+){0,2}(?:asked|required|charged|expected|request(?:ed)?)\b",
    re.IGNORECASE,
)


def _sentence_around(text: str, start: int, end: int) -> str:
    left = max(text.rfind(ch, 0, start) for ch in ".!?;\n") + 1
    rights = [i for i in (text.find(ch, end) for ch in ".!?;\n") if i != -1]
    return text[left:min(rights) if rights else len(text)]


def _category_match(text: str, rule: CategoryRule, negation_pattern: re.Pattern) -> str | None:
    """First match of a category rule that is not negated, a company statement, or (where relevant) after an offer."""
    for match in rule.pattern.finditer(text):
        if negation_pattern.search(text[max(0, match.start() - 60):match.start()]):
            continue
        sentence = _sentence_around(text, match.start(), match.end())
        if rule.negation_guard and _JOB_NEGATION_GUARD_RE.search(sentence):
            continue
        if rule.company_subject and COMPANY_SUBJECT_RE.search(text[max(0, match.start() - 20):match.start()]):
            continue
        if rule.extra_guard is not None and rule.extra_guard.search(sentence):
            continue
        return match.group(0)
    return None


def detect_warning_signs(text: str) -> list[dict]:
    """Return rule matches with matched text, excluding obvious negations."""
    findings = []
    negation_pattern = re.compile(
        r"\b(?:no|not|never|without|isn't|is not|"
        r"doesn't require|does not require|"
        r"do not require|don't require)\b"
        r"[^.!?;]{0,50}$",
        re.IGNORECASE,
    )

    for label, pattern in WARNING_RULES:
        for match in pattern.finditer(text):
            preceding_text = text[max(0, match.start() - 60):match.start()]

            # Avoid flagging a fee or request that is explicitly negated.
            if negation_pattern.search(preceding_text):
                continue

            findings.append({
                "label": label,
                "matched_text": match.group(0),
            })
            break

    # Category-level coverage: one sign per label, the base rules above win when both match.
    found = {finding["label"] for finding in findings}
    for rule in CATEGORY_RULES:
        if rule.label in found:
            continue
        matched = _category_match(text, rule, negation_pattern)
        if matched is not None:
            findings.append({"label": rule.label, "matched_text": matched})
            found.add(rule.label)

    return findings


def classifier_risk_score(fraud_probability: float, threshold: float) -> int:
    """Convert classifier probability to a provisional 0-100 score."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1 (exclusive)")

    p = min(max(float(fraud_probability), 0.0), 1.0)

    if p < threshold:
        score = p / threshold * 29
    else:
        score = 30 + (p - threshold) / (1.0 - threshold) * 70

    return int(round(min(100.0, max(0.0, score))))


def risk_category(score: int) -> str:
    """Provisional ranges: 0-29, 30-59, 60-100."""
    if score < 30:
        return "Lower risk"
    if score < 60:
        return "Moderate risk"
    return "Higher risk"


def combined_risk_score(
    classifier_score: int,
    warning_signs: list[dict],
) -> int:
    """
    Combine the classifier score with rule-based warning signs.

    This is a provisional heuristic, not a calibrated fraud probability.
    Each distinct rule-based warning category contributes 10 points,
    capped at 30 additional points.
    """
    distinct_labels = {
        str(sign.get("label", "")).strip().casefold()
        for sign in warning_signs
        if isinstance(sign, dict) and sign.get("label")
    }

    warning_adjustment = min(len(distinct_labels) * 10, 30)

    return min(100, max(0, int(classifier_score)) + warning_adjustment)


VERDICT_LEGITIMATE = "Likely Legitimate"
VERDICT_NEEDS_VERIFICATION = "Needs Verification"
VERDICT_HIGH_RISK = "High Fraud Risk"
VERDICT_INSUFFICIENT = "Insufficient Information"
INSUFFICIENT_MESSAGE = "Not enough text to assess. Add a description."


def insufficient_report() -> dict:
    """Report for a posting with too little text; the classifier is not run, so there is no score."""
    return {
        "verdict": VERDICT_INSUFFICIENT,
        "verdict_reason": INSUFFICIENT_MESSAGE,
        "risk_score": None,
        "risk_category": None,
        "display_risk_score": None,
        "classifier_risk_score": None,
        "warning_adjustment": None,
        "classifier_prediction": None,
        "classifier_details": None,
        "warning_signs": [],
        "supporting_evidence": [],
        "explanation": INSUFFICIENT_MESSAGE,
        "uncertainty_notes": ["The classifier was not run because the posting has too little text to assess."],
    }


def job_verdict(
    fraud_probability: float,
    threshold: float,
    company_profile: str | None = None,
) -> tuple[str, str]:
    """
    Return (verdict, reason).

    company_profile=None means the field was not provided; "" or whitespace means
    it was provided but empty. The profile only matters once the classifier flags
    the posting. Exploratory rule, chosen after inspecting validation errors.
    """
    if float(fraud_probability) < threshold:
        return VERDICT_LEGITIMATE, "fraud probability is below the decision threshold"
    if company_profile is None:
        return VERDICT_NEEDS_VERIFICATION, "company profile not provided"
    if not str(company_profile).strip():
        return VERDICT_NEEDS_VERIFICATION, (
            "classifier flagged the posting and the company profile is empty"
        )
    return VERDICT_HIGH_RISK, (
        "classifier flagged the posting even though a company profile is present"
    )


ESCALATION_MIN_CATEGORIES = 2
# Specific scam patterns that escalate on their own. Adopted after the pre-registered validation check: fired
# alone on 0 of 2,537 legitimate "Likely Legitimate" validation postings (bar: 10 or fewer).
SINGLE_ESCALATION_CATEGORIES = (
    "Cheque or overpayment scam",
    "Gift card or crypto payment",
    "Parcel or money forwarding",
)


def escalate_verdict(verdict: str, reason: str, warning_signs: list[dict]) -> tuple[str, str]:
    """
    Escalation only: a "Likely Legitimate" verdict becomes "Needs Verification" when
      * at least ESCALATION_MIN_CATEGORIES distinct rule categories fired (any kind), or
      * any one of SINGLE_ESCALATION_CATEGORIES fired, even alone.
    The reason names the category or categories that raised it. It never lowers a verdict and never touches a
    score, threshold or model output. The classifier-only rule (job_verdict) is unchanged; this wrapper sits on top.
    """
    if verdict != VERDICT_LEGITIMATE:
        return verdict, reason
    categories = list(dict.fromkeys(
        str(sign["label"]).strip() for sign in warning_signs
        if isinstance(sign, dict) and str(sign.get("label", "")).strip()
    ))
    if len(categories) >= ESCALATION_MIN_CATEGORIES:
        return VERDICT_NEEDS_VERIFICATION, f"scam-pattern indicators present ({', '.join(categories)})"
    specific = [c for c in categories if c in SINGLE_ESCALATION_CATEGORIES]
    if specific:
        return VERDICT_NEEDS_VERIFICATION, f"specific scam-pattern indicator present ({specific[0]})"
    return verdict, reason


def display_risk_score(fraud_logodds: float | None, threshold: float) -> int | None:
    """
    Display-only 1-99 score: 100 * sigmoid((logodds - logit(threshold)) / 3), truncated.

    50 sits exactly at the decision threshold. Not a probability. None without log-odds.
    """
    if fraud_logodds is None:
        return None
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1 (exclusive)")

    z = (float(fraud_logodds) - math.log(threshold / (1.0 - threshold))) / 3.0
    # Numerically stable sigmoid for very large |z|.
    if z >= 0:
        s = 1.0 / (1.0 + math.exp(-z))
    else:
        s = math.exp(z) / (1.0 + math.exp(z))

    return clamp_display_score(100.0 * s)


def build_job_report(
    prediction: dict,
    text: str = "",
    company_profile: str | None = None,
) -> dict:
    """Build the report from a DistilBERT prediction and the original posting."""
    probability = float(prediction["fraud_probability"])
    threshold = float(prediction["threshold"])
    flagged = int(prediction["prediction"]) == 1
    classifier_score = classifier_risk_score(probability, threshold)

    warning_signs = detect_warning_signs(text)

    score = combined_risk_score(classifier_score, warning_signs)

    if flagged:
        explanation = (
            "The DistilBERT classifier flagged this posting as potentially fraudulent. "
            "This is a pattern match on the wording, not proof about the company. "
            "Review the warning signs and verify the employer independently before "
            "sharing money or personal details."
        )
    else:
        explanation = (
            "The DistilBERT classifier did not flag this posting. "
            "That does not guarantee the job is genuine. Verify the employer independently."
        )

    verdict, verdict_reason = job_verdict(probability, threshold, company_profile)
    verdict, verdict_reason = escalate_verdict(verdict, verdict_reason, warning_signs)

    uncertainty_notes = [PROVISIONAL_SCORE_NOTE, MODEL_LIMITS_NOTE]

    if not warning_signs:
        uncertainty_notes.append(
            "No phrases matched the current warning-sign rules. This does not mean the posting is safe."
        )

    return {
        "risk_score": score,
        "risk_category": risk_category(score),
        "classifier_risk_score": classifier_score,
        "warning_adjustment": score - classifier_score,
        "classifier_prediction": (
            "Potentially fraudulent" if flagged else "Likely legitimate"
        ),
        "classifier_details": {
            "model": prediction.get("model", "DistilBERT"),
            "fraud_probability": round(probability, 6),
            "fraud_logodds": prediction.get("fraud_logodds"),
            "threshold": threshold,
        },
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "display_risk_score": display_risk_score(prediction.get("fraud_logodds"), threshold),
        "warning_signs": warning_signs,
        "supporting_evidence": [],
        "explanation": explanation,
        "uncertainty_notes": uncertainty_notes,
    }