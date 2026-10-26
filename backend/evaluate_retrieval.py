"""Evaluate semantic retrieval using Recall@k, Precision@k, and MRR."""

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.semantic_retrieval import retrieve_semantic

DATASET_PATH = PROJECT_ROOT / "backend" / "data" / "retrieval_eval.json"
K_VALUES = [3, 5, 10]


def load_evaluation_dataset() -> list[dict[str, Any]]:
    """Load and validate the retrieval evaluation dataset."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH}"
        )

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Evaluation dataset must be a non-empty JSON list.")

    for index, item in enumerate(dataset, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Evaluation item {index} must be a JSON object.")

        query = item.get("query")
        relevant_ids = item.get("relevant_source_ids")

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                f"Evaluation item {index} must have a non-empty query."
            )

        if (
            not isinstance(relevant_ids, list)
            or not relevant_ids
            or not all(isinstance(source_id, str) for source_id in relevant_ids)
        ):
            raise ValueError(
                f"Evaluation item {index} must have a non-empty "
                "'relevant_source_ids' list of strings."
            )

    return dataset


def evaluate_query(
    query: str,
    relevant_ids: set[str],
    k: int,
) -> dict[str, Any]:
    """Calculate retrieval metrics for one query."""
    results = retrieve_semantic(query, top_k=k)
    retrieved_ids = [item["id"] for item in results]

    relevant_retrieved = sum(
        1 for source_id in retrieved_ids if source_id in relevant_ids
    )

    # Recall: proportion of known relevant sources that were retrieved.
    recall = (
        relevant_retrieved / len(relevant_ids)
        if relevant_ids
        else 0.0
    )

    # Precision: proportion of returned results that are relevant.
    # Use the actual result count because the knowledge base may contain
    # fewer than k documents.
    precision = (
        relevant_retrieved / len(retrieved_ids)
        if retrieved_ids
        else 0.0
    )

    # Reciprocal rank: reciprocal of the position of the first relevant result.
    reciprocal_rank = 0.0

    for rank, source_id in enumerate(retrieved_ids, start=1):
        if source_id in relevant_ids:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "recall": recall,
        "precision": precision,
        "reciprocal_rank": reciprocal_rank,
        "retrieved_ids": retrieved_ids,
        "relevant_retrieved": relevant_retrieved,
    }


def main() -> None:
    """Run retrieval evaluation for each configured k value."""
    try:
        dataset = load_evaluation_dataset()

        # Check that the semantic index contains documents.
        indexed_sources = retrieve_semantic("job scam", top_k=10)

        if not indexed_sources:
            print("ERROR: No guidance sources found in the ChromaDB index.")
            print(
                "Rebuild the index before evaluating, using your existing "
                "rebuild_guidance_index() function."
            )
            sys.exit(1)

        print(f"Evaluation queries: {len(dataset)}")
        print(f"Sources returned by index check: {len(indexed_sources)}")

        for k in K_VALUES:
            scores = []

            for item in dataset:
                score = evaluate_query(
                    query=item["query"],
                    relevant_ids=set(item["relevant_source_ids"]),
                    k=k,
                )
                scores.append(score)

            count = len(scores)

            mean_recall = sum(
                score["recall"] for score in scores
            ) / count

            mean_precision = sum(
                score["precision"] for score in scores
            ) / count

            mean_mrr = sum(
                score["reciprocal_rank"] for score in scores
            ) / count

            print(f"\n--- Results for k={k} ---")
            print(f"Recall@{k}:    {mean_recall:.3f}")
            print(f"Precision@{k}: {mean_precision:.3f}")
            print(f"MRR:           {mean_mrr:.3f}")

    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"ERROR: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()