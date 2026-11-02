
"""Flask blueprint for job-posting analysis: DistilBERT, evidence retrieval and a grounded LLM explanation."""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from dotenv import load_dotenv
from flask import Blueprint, jsonify, request

from backend.services import llm_explainer
from backend.services.dataset import TEXT_FIELDS, build_text, clean_field
from backend.services.evidence import build_evidence
from backend.services.job_analysis import build_job_report, insufficient_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

job_bp = Blueprint("job_routes", __name__)
MAX_TEXT_CHARS = 20_000
MIN_POSTING_CHARS = 30  # same minimum the frontend applies to a pasted posting
BODY_FIELDS = ("description", "requirements", "benefits")

# Decided by the classifier and rules; nothing after build_job_report may change them.
FIXED_REPORT_KEYS = (
    "verdict", "verdict_reason", "risk_score", "risk_category",
    "display_risk_score", "classifier_risk_score", "classifier_prediction",
)

_classifier = None
_lock = threading.Lock()
_evidence_resources = None
_evidence_lock = threading.Lock()
_breaker = llm_explainer.CircuitBreaker()


def get_classifier():
    """Load the DistilBERT service once, on first use."""
    global _classifier

    with _lock:
        if _classifier is None:
            from backend.services.distilbert_service import DistilBERTJobScamService

            _classifier = DistilBERTJobScamService()

        return _classifier


def get_evidence_resources() -> dict:
    """Load the cached train-only posting index and entity graph once; missing caches become None."""
    global _evidence_resources

    with _evidence_lock:
        if _evidence_resources is None:
            from backend.services import entity_graph, posting_index

            resources = {"posting_index": None, "graph": None}
            try:
                train_sha = posting_index.manifest_train_sha256()
            except Exception:
                logger.warning("Split manifest not found; evidence indexes are unavailable.")
                train_sha = None
            if train_sha:
                for key, load in (("posting_index", posting_index.load_index),
                                  ("graph", entity_graph.load_graph)):
                    try:
                        resources[key] = load(train_sha)
                    except FileNotFoundError:
                        logger.warning("Evidence cache %s not built; run scripts/build_evidence_indexes.py.", key)
                    except Exception:
                        logger.exception("Could not load evidence cache %s.", key)
            _evidence_resources = resources

        return _evidence_resources


def default_guidance(text: str) -> list:
    """Trusted guidance from the local Chroma index, using only the locally cached model."""
    from backend.services.semantic_retrieval import retrieve_semantic

    return retrieve_semantic(text, top_k=3, local_files_only=True)


def get_guidance_fn():
    return default_guidance


def get_llm_client():
    return llm_explainer.client_from_env()


def is_structured_request(payload) -> bool:
    """True when the request carries posting fields beyond the company profile."""
    return isinstance(payload, dict) and any(
        field in payload for field in TEXT_FIELDS if field != "company_profile"
    )


def structured_posting(payload: dict) -> tuple[str | None, str | None]:
    """
    Build the classifier input exactly as the training text was built.

    Returns (text, company_profile). text is None if any field is not a string.
    company_profile is None when not provided and "" when provided but empty
    after the same HTML cleaning used for training.
    """
    fields = {field: payload.get(field) for field in TEXT_FIELDS}
    if any(value is not None and not isinstance(value, str) for value in fields.values()):
        return None, None

    company_profile = fields["company_profile"]
    if company_profile is not None:
        company_profile = clean_field(company_profile) or ""

    return build_text(fields), company_profile


def has_too_little_text(payload: dict) -> bool:
    """No description/requirements/benefits content, or under MIN_POSTING_CHARS of cleaned text in total."""
    cleaned = {field: clean_field(payload.get(field)) or "" for field in TEXT_FIELDS}
    has_body = any(cleaned[field] for field in BODY_FIELDS)
    total = len(" ".join(value for value in cleaned.values() if value))
    return not has_body or total < MIN_POSTING_CHARS


def add_evidence_and_explanation(report: dict, text: str, posting_fields: dict) -> dict:
    """Always attach evidence and an explanation (LLM or template); never change the fixed fields."""
    fixed = {key: report[key] for key in FIXED_REPORT_KEYS}
    resources = get_evidence_resources()

    evidence = build_evidence(
        text,
        posting_fields,
        report["warning_signs"],
        posting_index=resources.get("posting_index"),
        graph=resources.get("graph"),
        guidance_fn=get_guidance_fn(),
    )
    result = llm_explainer.explain(
        dict(fixed), evidence, text, client=get_llm_client(), breaker=_breaker,
    )

    report["evidence"] = evidence
    report["supporting_evidence"] = evidence["guidance"]
    report["explanation"] = result["explanation"]
    report["explanation_source"] = result["explanation_source"]
    report["cited_evidence_ids"] = result["cited_evidence_ids"]
    report["llm_caveats"] = result["caveats"]
    report["llm_status"] = result["llm_status"]
    if result["llm_status"] != "ok":
        # Status codes only; never the posting, the model output or provider error text.
        logger.warning("LLM explanation fell back to the template: status=%s detail=%s rate_limit=%s",
                       result["llm_status"], result["llm_detail"], result.get("rate_limit_kind") or "-")

    if any(report[key] != value for key, value in fixed.items()):
        logger.error("A fixed report field changed after the explanation step; restoring it.")
        report.update(fixed)
    return report


DEMO_EXAMPLES_PATH = PROJECT_ROOT / "backend" / "data" / "demo_examples.json"


@job_bp.get("/api/demo-examples")
def demo_examples_route():
    """Return the prepared example postings (the five posting fields plus a short note)."""
    try:
        data = json.loads(DEMO_EXAMPLES_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return jsonify([]), 200
    if not isinstance(data, list):
        return jsonify([]), 200
    examples = [{field: item.get(field) if isinstance(item.get(field), str) else ""
                 for field in (*TEXT_FIELDS, "note")}
                for item in data if isinstance(item, dict)]
    return jsonify(examples), 200


@job_bp.post("/api/analyze-job")
def analyze_job_route():
    """
    Accept JSON and return a report. Two request shapes:

    * {"text": "<job posting>", "company_profile": optional} -- the pasted text becomes the
      description and goes through the same path as the structured shape below, so the
      classifier sees the cleaned text and, when given, the cleaned company profile.
    * Structured: any of title, description, requirements, benefits (plus optional
      company_profile). The classifier input is built with the training text builder;
      "text" is ignored in this shape.

    Both shapes use structured_posting: the model text and the "profile present" check
    for the verdict come from the same cleaned profile (HTML-only or entity-only = empty).
    """
    payload = request.get_json(silent=True)
    structured = is_structured_request(payload)

    if not structured:
        pasted = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(pasted, str) or not pasted.strip():
            return jsonify({"error": "Please provide the job posting text."}), 400
        if len(pasted) > MAX_TEXT_CHARS:
            return jsonify({
                "error": (
                    f"The posting is too long. "
                    f"Please keep it under {MAX_TEXT_CHARS:,} characters."
                )
            }), 413
        profile = payload.get("company_profile")
        # Same fields a form would send: description = the pasted text, profile only if it is a string.
        payload = {"description": pasted, "company_profile": profile if isinstance(profile, str) else None}

    text, company_profile = structured_posting(payload)

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Please provide the job posting text."}), 400

    if len(text) > MAX_TEXT_CHARS:
        return jsonify({
            "error": (
                f"The posting is too long. "
                f"Please keep it under {MAX_TEXT_CHARS:,} characters."
            )
        }), 413

    if structured and has_too_little_text(payload):
        # Only a title/profile, or a few words: not enough to score. No classifier, evidence or LLM.
        return jsonify(insufficient_report()), 200

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

    # None means "not provided"; "" means provided but empty after cleaning (both reported in the verdict).
    posting_fields = {field: payload.get(field) for field in TEXT_FIELDS}

    # Pass the posting text so rule-based warning detection can run.
    report = build_job_report(prediction, text, company_profile)
    # New field only: True when the cleaned profile has any text (HTML-only or entity-only counts as none).
    report["company_profile_present"] = bool(company_profile and str(company_profile).strip())

    # Retrieval evidence and a grounded explanation; the verdict and scores stay fixed.
    report = add_evidence_and_explanation(report, text, posting_fields)

    return jsonify(report), 200
