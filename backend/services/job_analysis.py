
"""Turn a DistilBERT prediction into a report with job-posting warning signs."""
from __future__ import annotations

import re

PROVISIONAL_SCORE_NOTE = (
    "The score is provisional. It is the DistilBERT fraud probability rescaled so that the model's "
    "decision threshold sits at 30. The planned combined risk score is not built yet."
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
def detect_warning_signs(text: str) -> list[dict]:
    """Return rule matches with the matched text for display and review."""
    findings = []

    for label, pattern in WARNING_RULES:
        match = pattern.search(text)
        if match:
            findings.append({
                "label": label,
                "matched_text": match.group(0),
            })

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


def build_job_report(prediction: dict, text: str = "") -> dict:
    """Build the report from a DistilBERT prediction and the original posting."""
    probability = float(prediction["fraud_probability"])
    threshold = float(prediction["threshold"])
    flagged = int(prediction["prediction"]) == 1
    score = classifier_risk_score(probability, threshold)

    warning_signs = detect_warning_signs(text)

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

    uncertainty_notes = [PROVISIONAL_SCORE_NOTE, MODEL_LIMITS_NOTE]

    if not warning_signs:
        uncertainty_notes.append(
            "No phrases matched the current warning-sign rules. This does not mean the posting is safe."
        )

    return {
        "risk_score": score,
        "risk_category": risk_category(score),
        "classifier_prediction": (
            "Potentially fraudulent" if flagged else "Likely legitimate"
        ),
        "classifier_details": {
            "model": prediction.get("model", "DistilBERT"),
            "fraud_probability": round(probability, 6),
            "threshold": threshold,
        },
        "warning_signs": warning_signs,
        "supporting_evidence": [],
        "explanation": explanation,
        "uncertainty_notes": uncertainty_notes,
    }
