import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from services.knowledge_base import load_knowledge_base

DATA_PATH = PROJECT_ROOT / "backend" / "data"
GUIDANCE_PATH = DATA_PATH / "trusted_guidance_candidates.jsonl"
EVAL_PATH = DATA_PATH / "retrieval_eval.json"

MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
]
K_VALUES = [3, 5, 10]


def record_text(record):
    """Build searchable text from a guidance record."""
    fields = [
        record.title,
        record.publisher,
        record.excerpt,
        " ".join(record.warning_sign_tags),
    ]
    return " ".join(str(value) for value in fields if value)


def main():
    records = load_knowledge_base(GUIDANCE_PATH)

    with EVAL_PATH.open("r", encoding="utf-8") as file:
        queries = json.load(file)

    if not records:
        raise ValueError("No guidance records were loaded.")

    record_ids = [record.source_id for record in records]
    texts = [record_text(record) for record in records]

    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Duplicate source IDs found in guidance data.")

    print(f"Guidance sources: {len(records)}")
    print(f"Evaluation queries: {len(queries)}")

    for model_name in MODELS:
        print(f"\nLoading model: {model_name}")
        model = SentenceTransformer(model_name)

        document_vectors = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        totals = {
            k: {"recall": 0.0, "precision": 0.0}
            for k in K_VALUES
        }
        reciprocal_rank_total = 0.0

        query_vectors = model.encode(
            [item["query"] for item in queries],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        for item, query_vector in zip(queries, query_vectors):
            # Normalized embeddings let us use dot product for cosine similarity.
            scores = document_vectors @ query_vector
            ranked_indices = np.argsort(-scores)
            ranked_ids = [record_ids[i] for i in ranked_indices]

            relevant = set(item["relevant_source_ids"])
            if not relevant:
                continue

            for k in K_VALUES:
                returned = ranked_ids[:k]
                hits = sum(source_id in relevant for source_id in returned)

                totals[k]["recall"] += hits / len(relevant)
                totals[k]["precision"] += hits / len(returned) if returned else 0.0

            first_relevant_rank = next(
                (
                    rank
                    for rank, source_id in enumerate(ranked_ids, start=1)
                    if source_id in relevant
                ),
                None,
            )

            if first_relevant_rank is not None:
                reciprocal_rank_total += 1 / first_relevant_rank

        count = len(queries)
        print(f"\nResults for {model_name}:")

        for k in K_VALUES:
            print(
                f"Recall@{k}:    {totals[k]['recall'] / count:.3f} | "
                f"Precision@{k}: {totals[k]['precision'] / count:.3f}"
            )

        print(f"MRR:           {reciprocal_rank_total / count:.3f}")

    print("\nComparison complete. Your existing ChromaDB index was not modified.")


if __name__ == "__main__":
    main()