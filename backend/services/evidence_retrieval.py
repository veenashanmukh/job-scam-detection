
"""Retrieve relevant evidence from the validated trusted-guidance knowledge base.

This module performs local lexical retrieval only. It does not call an LLM,
fetch URLs, or determine whether a job posting is fraudulent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.services.knowledge_base import KnowledgeBaseRecord


@dataclass(frozen=True)
class RetrievedEvidence:
    """A knowledge-base record and its relevance score."""

    record: KnowledgeBaseRecord
    relevance_score: float

    def to_dict(self) -> dict:
        """Return evidence in a JSON-friendly format."""
        return {
            "source_id": self.record.source_id,
            "publisher": self.record.publisher,
            "title": self.record.title,
            "url": self.record.url,
            "retrieved_date": self.record.retrieved_date.isoformat(),
            "excerpt": self.record.excerpt,
            "warning_sign_tags": list(self.record.warning_sign_tags),
            "relevance_score": round(self.relevance_score, 4),
        }


def _document_text(record: KnowledgeBaseRecord) -> str:
    """Combine searchable fields from a trusted-guidance record."""
    return " ".join(
        (
            record.title,
            record.publisher,
            record.excerpt,
            " ".join(record.warning_sign_tags),
        )
    )


def retrieve_evidence(
    query: str,
    records: tuple[KnowledgeBaseRecord, ...] | list[KnowledgeBaseRecord],
    *,
    top_k: int = 3,
    min_score: float = 0.0,
) -> list[RetrievedEvidence]:
    """Rank trusted-guidance records by TF-IDF cosine similarity.

    Args:
        query: Job-posting or conversation text to search for.
        records: Previously validated knowledge-base records.
        top_k: Maximum number of results to return.
        min_score: Minimum cosine similarity required, from 0 to 1.

    Returns:
        Evidence sorted from highest to lowest relevance.

    Notes:
        A relevance score measures text similarity, not scam probability,
        source reliability, or proof that a claim is true.
    """
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if not isinstance(min_score, (int, float)) or isinstance(min_score, bool):
        raise TypeError("min_score must be a number")
    if not 0.0 <= min_score <= 1.0:
        raise ValueError("min_score must be between 0 and 1")

    query = query.strip()
    if not query or not records:
        return []

    documents = [_document_text(record) for record in records]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w+\b",
    )

    try:
        document_vectors = vectorizer.fit_transform(documents)
        query_vector = vectorizer.transform([query])
    except ValueError:
        # For example, every document and the query may contain only
        # words that the vectorizer cannot tokenize.
        return []

    scores = cosine_similarity(query_vector, document_vectors).ravel()

    ranked_indices = sorted(
        range(len(records)),
        key=lambda index: (-float(scores[index]), index),
    )

    results = []
    for index in ranked_indices:
        score = float(scores[index])
        if score < min_score:
            continue
        if score <= 0.0:
            continue

        results.append(
            RetrievedEvidence(
                record=records[index],
                relevance_score=score,
            )
        )
        if len(results) >= top_k:
            break

    return results
