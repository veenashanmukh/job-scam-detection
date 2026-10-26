"""Flask blueprint for job-posting analysis using DistilBERT and optional LLM + RAG."""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from dotenv import load_dotenv

from flask import Blueprint, jsonify, request

from backend.services.job_analysis import build_job_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

job_bp = Blueprint("job_routes", __name__)
MAX_TEXT_CHARS = 20_000

_classifier = None
_lock = threading.Lock()


def get_classifier():
    """Load the DistilBERT service once, on first use."""
    global _classifier

    with _lock:
        if _classifier is None:
            from backend.services.distilbert_service import DistilBERTJobScamService

            _classifier = DistilBERTJobScamService()

        return _classifier


def add_llm_analysis(report: dict, text: str) -> dict:
    """Add evidence-grounded LLM analysis without changing classifier results."""
    if os.getenv("ENABLE_LLM_ANALYSIS", "false").strip().lower() not in {
        "1", "true", "yes", "on"
    }:
        return report

    try:
        from backend.services.llm_analysis import analyze

        result = analyze(text)
        analysis = result["analysis"]

        # Preserve the original DistilBERT score and prediction.
        report["llm_analysis"] = analysis
        report["warning_signs"] = analysis.get("warning_signs", [])
        report["supporting_evidence"] = result.get("supporting_evidence", [])
        report["explanation"] = analysis.get("explanation", report["explanation"])

        # Remove the note saying RAG is not connected now that it succeeded.
        report["uncertainty_notes"] = [
            note
            for note in report["uncertainty_notes"]
            if "not connected for job postings yet" not in note.lower()
        ]

        # Preserve existing limitations and add the LLM's limitations.
        for limitation in analysis.get("limitations", []):
            if limitation not in report["uncertainty_notes"]:
                report["uncertainty_notes"].append(limitation)

        report["recommended_actions"] = analysis.get("recommended_actions", [])
        report["llm_model"] = result.get("model")

    except Exception:
        # Keep the classifier report available if LLM or retrieval fails.
        logger.warning(
            "LLM job analysis was unavailable; returning classifier-only report.",
            exc_info=True,
        )
        report["llm_analysis_status"] = "unavailable"
        report["uncertainty_notes"].append(
            "Additional LLM analysis or evidence retrieval was unavailable. "
            "The displayed classifier result is provided without that additional analysis."
        )
    else:
        report["llm_analysis_status"] = "success"

    return report


@job_bp.post("/api/analyze-job")
def analyze_job_route():
    """Accept JSON {"text": "<job posting>"} and return an analysis report."""
    payload = request.get_json(silent=True)
    text = payload.get("text") if isinstance(payload, dict) else None

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Please provide the job posting text."}), 400

    if len(text) > MAX_TEXT_CHARS:
        return jsonify({
            "error": (
                f"The posting is too long. "
                f"Please keep it under {MAX_TEXT_CHARS:,} characters."
            )
        }), 413

    try:
        classifier = get_classifier()
    except FileNotFoundError:
        logger.error("DistilBERT model files were not found.")
        return jsonify({
            "error": "The classifier model is not installed on this server."
        }), 503
    except Exception:
        logger.exception("Could not load the DistilBERT classifier.")
        return jsonify({
            "error": "The classifier could not be loaded."
        }), 503

    try:
        prediction = classifier.predict(text)
    except ValueError:
        return jsonify({"error": "Please provide the job posting text."}), 400
    except Exception:
        # Never log the submitted job-posting text.
        logger.exception("DistilBERT prediction failed.")
        return jsonify({"error": "Analysis failed. Please try again."}), 500

    report = build_job_report(prediction)
    report = add_llm_analysis(report, text)

    return jsonify(report), 200