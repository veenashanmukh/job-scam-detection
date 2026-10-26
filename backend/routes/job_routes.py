
"""Flask blueprint for single job-posting analysis with DistilBERT."""
from __future__ import annotations

import logging
import threading

from flask import Blueprint, jsonify, request

from backend.services.job_analysis import build_job_report
from backend.services.semantic_retrieval import retrieve_semantic

logger = logging.getLogger(__name__)

job_bp = Blueprint("job_routes", __name__)

MAX_TEXT_CHARS = 20_000

_classifier = None
_lock = threading.Lock()


def get_classifier():
    """Load the DistilBERT service once, on first use.

    The import is inside the function so chat API routes and unit tests
    do not need to load PyTorch unnecessarily.
    """
    global _classifier

    with _lock:
        if _classifier is None:
            from backend.services.distilbert_service import (
                DistilBERTJobScamService,
            )

            _classifier = DistilBERTJobScamService()

        return _classifier


@job_bp.post("/api/analyze-job")
def analyze_job_route():
    """Accept JSON {"text": "<job posting>"} and return an analysis report."""
    payload = request.get_json(silent=True)

    text = payload.get("text") if isinstance(payload, dict) else None

    # Validate the submitted job posting.
    if not isinstance(text, str) or not text.strip():
        return jsonify({
            "error": "Please provide the job posting text."
        }), 400

    if len(text) > MAX_TEXT_CHARS:
        return jsonify({
            "error": (
                f"The posting is too long. "
                f"Please keep it under {MAX_TEXT_CHARS:,} characters."
            )
        }), 413

    # Load the classifier.
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

    # Predict whether the posting may be fraudulent.
    try:
        prediction = classifier.predict(text)

    except ValueError:
        return jsonify({
            "error": "Please provide the job posting text."
        }), 400

    except Exception:
        # Never include the submitted job text in error logs.
        logger.exception("DistilBERT prediction failed.")

        return jsonify({
            "error": "Analysis failed. Please try again."
        }), 500

    # Build the report using both the classifier prediction and original text.
    # The report builder uses the text to detect configured warning phrases.
       # Build the initial report from the classifier prediction and job text.
    try:
        report = build_job_report(prediction, text)

    except Exception:
        logger.exception("Could not build the job analysis report.")

        return jsonify({
            "error": "Could not generate the analysis report."
        }), 500

    # Retrieve relevant trusted guidance for the submitted job posting.
    try:
        evidence = retrieve_semantic(text, top_k=3)
        report["supporting_evidence"] = evidence

    except Exception:
        # Keep classification available even if evidence retrieval fails.
        logger.exception("Semantic evidence retrieval failed.")
        report["supporting_evidence"] = []
        report["evidence_retrieval_status"] = "unavailable"

    else:
        report["evidence_retrieval_status"] = "success"

    return jsonify(report), 200