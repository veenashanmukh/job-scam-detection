"""Flask blueprint for plain-text WhatsApp/Telegram chat analysis."""
from __future__ import annotations

import copy
import logging

from flask import Blueprint, jsonify, request
from werkzeug.exceptions import RequestEntityTooLarge

from backend.routes import job_routes
from backend.services import llm_explainer
from backend.services.conversation_analysis import analyze_messages, redact_personal_data
from backend.services.conversation_parser import parse_chat_text, parse_untimestamped_conversation
from backend.services.evidence import build_evidence
from backend.services.text_display import shorten_quote

logger = logging.getLogger(__name__)

chat_bp = Blueprint("chat_routes", __name__)
MAX_CHAT_BYTES = 1 * 1024 * 1024  # 1 MiB

# Rule-based results; the explanation step must never change them.
FIXED_CHAT_KEYS = ("risk_score", "display_risk_score", "risk_category", "classifier_prediction", "warning_signs", "timeline")
# The explainer's consistency check uses the job-verdict stance matching each chat category.
CATEGORY_STANCE = {"Higher risk": "High Fraud Risk", "Moderate risk": "Needs Verification",
                   "Lower risk": "Likely Legitimate"}


@chat_bp.errorhandler(RequestEntityTooLarge)
def too_large(_error):
    return jsonify({"error": "Chat file is too large. Please upload a .txt file under 1 MiB."}), 413


MAX_FIRST_EXPLANATION_CHARS = 4000
RULE_QUOTE_CHARS = 120  # a rule sign's quote from the chat, cut at a word boundary
REUSED_NOTE = "(Explanation based on the first analysis; only the rule score was recomputed for the chosen name.)"


def add_chat_evidence_and_explanation(report: dict, messages: list, first_explanation: str = "") -> dict:
    """Guidance (K), rule signs (R) and flagged messages (T) from the other party, plus an explanation."""
    you = report.get("you")
    other_text = "\n".join(
        redact_personal_data(m["text"]) for m in messages
        if not m.get("is_system", False) and m["speaker"] != you
    )
    flagged = [t for t in report["timeline"] if t["sign"] and not t.get("own")]
    for i, entry in enumerate(flagged, 1):
        entry["evidence_id"] = f"T{i}"  # same numbering as the evidence step's messages
    fixed = copy.deepcopy({key: report[key] for key in FIXED_CHAT_KEYS})
    first_match = {}
    for entry in flagged:
        for label in entry["sign"].split("; "):
            first_match.setdefault(label, shorten_quote(entry["text"], RULE_QUOTE_CHARS))
    rule_signs = [{"label": label, "matched_text": first_match.get(label, "")} for label in report["warning_signs"]]

    evidence = build_evidence(
        other_text, {}, rule_signs,
        guidance_fn=job_routes.get_guidance_fn() if other_text.strip() else None,
        flagged_messages=flagged,
    )
    # No posting index or entity graph for chats: they hold job postings, not conversations.
    evidence.pop("similar_postings"), evidence.pop("similar_status"), evidence.pop("similar_min_similarity")
    evidence.pop("graph"), evidence.pop("graph_status")

    facts = {"verdict": f"{report['risk_category']} conversation",
             "verdict_reason": report["classifier_prediction"],
             "display_risk_score": report["display_risk_score"],
             "fact_label": "Assessment", "subject": "conversation"}
    if first_explanation:
        # Re-check with a chosen name: reuse the first explanation, no second LLM call.
        result = {"explanation": f"{first_explanation[:MAX_FIRST_EXPLANATION_CHARS]}\n\n{REUSED_NOTE}",
                  "explanation_source": "first_analysis", "cited_evidence_ids": [], "caveats": [],
                  "llm_status": "reused", "llm_detail": ""}
    else:
        result = llm_explainer.explain(
            facts, evidence, other_text, client=job_routes.get_llm_client(), breaker=job_routes._breaker,
            consistency_verdict=CATEGORY_STANCE[report["risk_category"]],
        )

    report["evidence"] = evidence
    report["supporting_evidence"] = evidence["guidance"]
    report["explanation"] = result["explanation"]
    report["explanation_source"] = result["explanation_source"]
    report["cited_evidence_ids"] = result["cited_evidence_ids"]
    report["llm_caveats"] = result["caveats"]
    report["llm_status"] = result["llm_status"]

    if result["llm_status"] != "ok":
        logger.warning("Chat explanation fell back to the template: status=%s detail=%s rate_limit=%s",
                       result["llm_status"], result["llm_detail"], result.get("rate_limit_kind") or "-")
    if any(report[key] != value for key, value in fixed.items()):
        logger.error("A fixed chat field changed after the explanation step; restoring it.")
        report.update(fixed)
    return report


@chat_bp.post("/api/analyze-chat")
def analyze_chat_route():
    """Accept multipart/form-data with a .txt file in field 'file' and an optional 'you' sender name."""
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"error": "Please upload a chat export as a .txt file."}), 400
    if not upload.filename.lower().endswith(".txt"):
        return jsonify({"error": "Unsupported file type. Please upload a plain-text .txt export."}), 400
    raw = upload.read(MAX_CHAT_BYTES + 1)
    if len(raw) > MAX_CHAT_BYTES:
        return jsonify({"error": "Chat file is too large. Please upload a .txt file under 1 MiB."}), 413
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return jsonify({"error": "Could not read this file as UTF-8 text. Re-export it as plain text."}), 400
    messages = parse_chat_text(text)
    if not messages and request.form.get("untimestamped") == "1":
        # Opt-in for a pasted "Speaker: text" conversation; uploads of real exports keep the strict parser.
        messages = parse_untimestamped_conversation(text)
    if not messages:
        return jsonify({"error": "No supported timestamped messages were found in this file."}), 400
    # Do not persist the upload or include raw chat content in logs.
    you = request.form.get("you") or None
    # Only a re-check (a chosen name) may reuse an earlier explanation.
    first_explanation = (request.form.get("first_explanation") or "").strip() if you else ""
    report = analyze_messages(messages, self_name=you)
    return jsonify(add_chat_evidence_and_explanation(report, messages, first_explanation)), 200
