
"""Flask blueprint for job-posting analysis using DistilBERT and optional LLM + RAG."""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from dotenv import load_dotenv
from flask import Blueprint, jsonify, request

from backend.services.job_analysis import build_job_report
import re

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


def normalize_warning_category(value: str) -> str:
    """Normalize equivalent rule-based labels and LLM warning identifiers."""
    normalized = str(value).strip().casefold()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")

    aliases = {
        "upfront_payment": "upfront_payment",
        "upfront_fee": "upfront_payment",
        "registration_fee": "upfront_payment",
        "pressure_tactic": "pressure_tactic",
        "pressure": "pressure_tactic",
        "requests_personal_information": "sensitive_information",
        "request_for_sensitive_info": "sensitive_information",
        "sensitive_information": "sensitive_information",
        "off_platform_messaging": "off_platform_messaging",
        "unrealistic_earnings": "unrealistic_earnings",
        "no_interview": "no_interview",
    }
    return aliases.get(normalized, normalized)


def combine_warning_signs(rule_signs: list, llm_signs: list) -> list:
    """Combine warnings, preserving rule-based evidence and removing semantic duplicates."""
    combined = []
    seen_categories = set()

    # Process rule-based findings first so matched text is preserved.
    for sign in rule_signs:
        if isinstance(sign, dict):
            label = str(sign.get("label", "")).strip()
            category = normalize_warning_category(label)
            key = category or repr(sign)

            if key not in seen_categories:
                combined.append(sign)
                seen_categories.add(key)
        else:
            category = normalize_warning_category(sign)
            key = category or repr(sign)

            if key not in seen_categories:
                combined.append(sign)
                seen_categories.add(key)

    # Add LLM findings only when their category is not already represented.
    for sign in llm_signs:
        if isinstance(sign, dict):
            label = str(sign.get("label", sign.get("category", ""))).strip()
        else:
            label = str(sign).strip()

        category = normalize_warning_category(label)
        key = category or repr(sign)

        if key not in seen_categories:
            combined.append(sign)
            seen_categories.add(key)

    return combined

def add_llm_analysis(report: dict, text: str) -> dict:
    """Add optional evidence-grounded LLM analysis without changing classifier results."""
    enabled = os.getenv(
        "ENABLE_LLM_ANALYSIS", "false"
    ).strip().lower() in {"1", "true", "yes", "on"}

    if not enabled:
        report["llm_analysis_status"] = "disabled"
        report["uncertainty_notes"] = [
            note for note in report["uncertainty_notes"]
            if "not connected for job postings yet" not in note.lower()
        ]
        report["uncertainty_notes"].append(
            "Additional LLM analysis and retrieved supporting evidence "
            "were not included because the feature is disabled."
        )
        return report

    try:
        from backend.services.llm_analysis import analyze

        result = analyze(text)
        analysis = result["analysis"]

        report["llm_analysis"] = analysis

        # Keep rule-based findings and add the LLM's findings.
        report["warning_signs"] = combine_warning_signs(
            report.get("warning_signs", []),
            analysis.get("warning_signs", []),
        )

        report["supporting_evidence"] = result["supporting_evidence"]
        report["explanation"] = analysis["explanation"]
        report["recommended_actions"] = analysis["recommended_actions"]
        report["llm_model"] = result["model"]

        report["uncertainty_notes"] = [
            note for note in report["uncertainty_notes"]
            if "not connected for job postings yet" not in note.lower()
        ]

        for limitation in analysis["limitations"]:
            if limitation not in report["uncertainty_notes"]:
                report["uncertainty_notes"].append(limitation)

    except Exception:
        logger.warning(
            "LLM job analysis was unavailable; returning classifier-only report.",
            exc_info=True,
        )
        report["llm_analysis_status"] = "unavailable"
        report["uncertainty_notes"].append(
            "Additional LLM analysis and retrieved supporting evidence "
            "were unavailable. The classifier result is shown independently."
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

    # Pass the posting text so rule-based warning detection can run.
    report = build_job_report(prediction, text)

    # Add optional RAG + LLM analysis without discarding rule-based findings.
    report = add_llm_analysis(report, text)

    return jsonify(report), 200
