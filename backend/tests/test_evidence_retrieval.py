
from datetime import date

import pytest

from backend.services.evidence_retrieval import retrieve_evidence
from backend.services.knowledge_base import validate_record


def make_record(
    source_id: str,
    title: str,
    excerpt: str,
    tags: list[str],
):
    """Build a validated knowledge-base record for testing."""
    return validate_record(
        {
            "source_id": source_id,
            "publisher": "Test Employment Authority",
            "title": title,
            "url": f"https://example.org/{source_id}",
            "retrieved_date": date.today().isoformat(),
            "license_note": "Synthetic test data; not a real source.",
            "excerpt": excerpt,
            "warning_sign_tags": tags,
        }
    )


@pytest.fixture
def records():
    return [
        make_record(
            "payment-guide",
            "Job recruitment payment warning",
            "Legitimate employers should not demand an upfront recruitment fee "
            "or a refundable security deposit to guarantee employment.",
            ["payment_request", "upfront_fee"],
        ),
        make_record(
            "identity-guide",
            "Protect personal information",
            "Be careful when recruiters request bank account details, identity "
            "documents, or other sensitive personal information.",
            ["sensitive_information", "identity_theft"],
        ),
        make_record(
            "interview-guide",
            "Preparing for job interviews",
            "Research the employer, prepare examples of your skills, and "
            "practice answering common interview questions.",
            ["interview", "job_search"],
        ),
    ]


def test_retrieves_relevant_evidence(records):
    results = retrieve_evidence(
        "The recruiter wants an upfront recruitment fee before I can get the job.",
        records,
    )

    assert results
    assert results[0].record.source_id == "payment-guide"


def test_results_are_ranked_by_relevance(records):
    results = retrieve_evidence(
        "recruitment fee security deposit",
        records,
    )

    scores = [result.relevance_score for result in results]
    assert scores == sorted(scores, reverse=True)


def test_empty_query_returns_no_results(records):
    assert retrieve_evidence("   ", records) == []


def test_empty_knowledge_base_returns_no_results():
    assert retrieve_evidence("upfront recruitment fee", []) == []


def test_unrelated_query_returns_no_results(records):
    assert retrieve_evidence("volcano astronomy galaxies", records) == []


def test_top_k_limits_results(records):
    results = retrieve_evidence(
        "job recruiter employment information",
        records,
        top_k=1,
    )

    assert len(results) <= 1


def test_min_score_filters_weak_matches(records):
    results = retrieve_evidence(
        "recruitment fee",
        records,
        min_score=0.99,
    )

    assert all(result.relevance_score >= 0.99 for result in results)


def test_scores_are_between_zero_and_one(records):
    results = retrieve_evidence("upfront recruitment fee", records)

    assert results
    assert all(0.0 < result.relevance_score <= 1.0 for result in results)


def test_result_includes_source_attribution(records):
    results = retrieve_evidence("upfront recruitment fee", records)

    evidence = results[0].to_dict()

    assert evidence["source_id"] == "payment-guide"
    assert evidence["publisher"] == "Test Employment Authority"
    assert evidence["title"] == "Job recruitment payment warning"
    assert evidence["url"] == "https://example.org/payment-guide"
    assert evidence["excerpt"]
    assert evidence["warning_sign_tags"] == ["payment_request", "upfront_fee"]
    assert 0.0 < evidence["relevance_score"] <= 1.0


def test_invalid_top_k_is_rejected(records):
    with pytest.raises(ValueError):
        retrieve_evidence("recruitment fee", records, top_k=0)


def test_invalid_min_score_is_rejected(records):
    with pytest.raises(ValueError):
        retrieve_evidence("recruitment fee", records, min_score=1.5)
