"""Turn a DistilBERT prediction into the shared API report for a job posting.

This module does not load a model. It only converts the output of
``DistilBERTJobScamService.predict`` into the report format in
``backend/schemas/analysis_response.md``, so it can be tested without torch.
"""
from __future__ import annotations

PROVISIONAL_SCORE_NOTE = (
    "The score is provisional. It is the DistilBERT fraud probability rescaled so that the model's "
    "decision threshold sits at 30. The planned combined risk score is not built yet."
)
NOT_CONNECTED_NOTE = (
    "Warning-sign rules and evidence retrieval are not connected for job postings yet, "
    "so no warning signs or supporting evidence are shown."
)
MODEL_LIMITS_NOTE = (
    "The classifier can miss some scams and can flag some legitimate postings. "
    "It reads the text only and does not check the company."
)


def classifier_risk_score(fraud_probability: float, threshold: float) -> int:
    """Rescale a fraud probability to 0-100 so the category always agrees with the classifier decision.

    Below the threshold the score runs from 0 to 29 (Lower risk). At or above it, the score runs from 30 to 100.
    This is a placeholder until the weighted-indicator risk score exists. It is not a probability.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1 (exclusive)")
    p = min(max(float(fraud_probability), 0.0), 1.0)
    if p < threshold:
        score = p / threshold * 29
    else:
        score = 30 + (p - threshold) / (1.0 - threshold) * 70
    return int(round(min(100.0, max(0.0, score))))


def risk_category(score: int) -> str:
    """Provisional ranges from the project plan: 0-29, 30-59, 60-100."""
    if score < 30:
        return "Lower risk"
    if score < 60:
        return "Moderate risk"
    return "Higher risk"


def build_job_report(prediction: dict) -> dict:
    """Build the API report from a DistilBERT prediction dictionary."""
    probability = float(prediction["fraud_probability"])
    threshold = float(prediction["threshold"])
    flagged = int(prediction["prediction"]) == 1
    score = classifier_risk_score(probability, threshold)

    if flagged:
        explanation = (
            "The DistilBERT classifier flagged this posting as potentially fraudulent. "
            "That is a pattern match on the wording, not proof about the company. "
            "Check the employer on its official website before sharing money or personal details."
        )
    else:
        explanation = (
            "The DistilBERT classifier did not flag this posting. "
            "That does not guarantee the job is genuine. Verify the employer independently."
        )

    return {
        "risk_score": score,
        "risk_category": risk_category(score),
        "classifier_prediction": "Potentially fraudulent" if flagged else "Likely legitimate",
        "classifier_details": {
            "model": prediction.get("model", "DistilBERT"),
            "fraud_probability": round(probability, 6),
            "threshold": threshold,
        },
        "warning_signs": [],
        "supporting_evidence": [],
        "explanation": explanation,
        "uncertainty_notes": [PROVISIONAL_SCORE_NOTE, NOT_CONNECTED_NOTE, MODEL_LIMITS_NOTE],
    }
