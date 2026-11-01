"""Evidence step for every job analysis: retrieval only, no LLM, no network.

Collects similar TRAIN postings (S1..), entity-graph links (G1..), trusted guidance (K1..)
and the rule-based warning signs (R1..). Each item gets a stable id that the LLM must cite.
A missing or failing source is reported as a status and never fails the request.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Mapping

from backend.services import entity_graph as eg
from backend.services import posting_index as pi
from backend.services.text_display import shorten_quote

logger = logging.getLogger(__name__)

TOP_SIMILAR = 5
MIN_SIMILARITY = 0.30  # weaker matches are not evidence and are omitted
TOP_GUIDANCE = 3
MIN_GUIDANCE_SIMILARITY = 0.25  # provisional; to be tuned. Weaker guidance is omitted and never cited
GUIDANCE_EXCERPT_CHARS = 300

NO_GRAPH_LINKS_NOTE = "No linked companies or contacts found."
KIND_NAMES = {"EMAIL": "email address", "PHONE": "phone number", "URL": "website", "COMPANY": "company profile"}


def _similar(index, text):
    if index is None:
        return [], "unavailable"
    try:
        hits = pi.search(index, text, top_k=TOP_SIMILAR)
    except Exception:
        logger.exception("Similar-posting retrieval failed.")
        return [], "error"
    close = [hit for hit in hits if hit["similarity"] >= MIN_SIMILARITY]
    return [{"id": f"S{i}", **hit} for i, hit in enumerate(close, 1)], "ok"


def _graph_message(kind_name: str, linked: int, fraud: int) -> str:
    return (f"This {kind_name} appears in {linked} known posting{'s' if linked != 1 else ''} "
            f"({fraud} labelled fraud).")


def _graph(graph, posting_fields):
    if graph is None:
        return {"items": []}, "unavailable"
    try:
        result = eg.query(graph, posting_fields)
    except Exception:
        logger.exception("Entity-graph query failed.")
        return {"items": []}, "error"

    # One item per matched entity, each with its own true totals (a posting shared by several entities counts under
    # each); linked_postings is only the short example list.
    items = []
    for i, link in enumerate(result["entities"], 1):
        kind = link["kind"]
        kind_name = KIND_NAMES.get(kind, kind.lower())
        items.append({
            "id": f"G{i}",
            "entity": link["entity"],  # hashed id only
            "kind": kind,
            "kind_name": kind_name,
            "linked_count": link["linked_total"],
            "fraud_count": link["fraud_total"],
            "linked_postings": link["examples"],
            "message": _graph_message(kind_name, link["linked_total"], link["fraud_total"]),
        })
    return {
        "items": items,
        "note": "" if items else NO_GRAPH_LINKS_NOTE,
        "entities_found": len(result["entities_found"]),
        "linked_postings_total": result["linked_postings_total"],
        "fraud_neighbour_count": result["fraud_neighbour_count"],
        "fraud_rate_smoothed": result["fraud_rate_smoothed"],
    }, "ok"


def _guidance(guidance_fn, text):
    if guidance_fn is None:
        return [], "unavailable"
    try:
        records = guidance_fn(text) or []
    except Exception:
        logger.warning("Guidance retrieval unavailable.", exc_info=True)
        return [], "unavailable"
    related = [
        record for record in records[:TOP_GUIDANCE]
        if isinstance(record.get("similarity"), (int, float)) and record["similarity"] >= MIN_GUIDANCE_SIMILARITY
    ]
    items = []
    for i, record in enumerate(related, 1):
        items.append({
            "id": f"K{i}",
            "source_id": record.get("id", ""),
            "title": record.get("title", ""),
            "publisher": record.get("publisher", ""),
            "url": record.get("url", ""),
            "excerpt": str(record.get("excerpt", ""))[:GUIDANCE_EXCERPT_CHARS],
            "warning_sign_tags": list(record.get("warning_sign_tags", [])),
            "similarity": record.get("similarity"),
        })
    return items, "ok"


MESSAGE_EXCERPT_CHARS = 160


def _messages(flagged: list) -> list[dict]:
    """Flagged chat messages (already redacted) as citable items T1.."""
    items = []
    for i, entry in enumerate(flagged, 1):
        text = str(entry.get("text", ""))
        items.append({
            "id": f"T{i}",
            "time": entry.get("time", ""),
            "speaker": entry.get("speaker", ""),
            "sign": entry.get("sign", ""),
            "text": shorten_quote(text, MESSAGE_EXCERPT_CHARS),
        })
    return items


def build_evidence(
    text: str,
    posting_fields: Mapping,
    rule_signs: list,
    *,
    posting_index=None,
    graph=None,
    guidance_fn: Callable[[str], list] | None = None,
    flagged_messages: list | None = None,
) -> dict:
    """Return the evidence block; sources passed as None are reported as unavailable."""
    similar, similar_status = _similar(posting_index, text)
    graph_block, graph_status = _graph(graph, posting_fields)
    guidance, guidance_status = _guidance(guidance_fn, text)
    rules = [
        {"id": f"R{i}", "label": sign.get("label", ""), "matched_text": sign.get("matched_text", "")}
        for i, sign in enumerate((s for s in rule_signs if isinstance(s, dict)), 1)
    ]
    return {
        "similar_postings": similar,
        "similar_status": similar_status,
        "similar_min_similarity": MIN_SIMILARITY,
        "graph": graph_block,
        "graph_status": graph_status,
        "guidance": guidance,
        "guidance_status": guidance_status,
        "guidance_min_similarity": MIN_GUIDANCE_SIMILARITY,
        "rule_signs": rules,
        "messages": _messages(flagged_messages or []),
    }


def evidence_ids(evidence: Mapping) -> list[str]:
    """Every citable id, in display order."""
    return (
        [item["id"] for item in evidence.get("similar_postings", [])]
        + [item["id"] for item in evidence.get("graph", {}).get("items", [])]
        + [item["id"] for item in evidence.get("guidance", [])]
        + [item["id"] for item in evidence.get("rule_signs", [])]
        + [item["id"] for item in evidence.get("messages", [])]
    )
