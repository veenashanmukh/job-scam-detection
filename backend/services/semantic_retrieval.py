"""Semantic retrieval over trusted job-scam guidance using ChromaDB."""

from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from backend.services.knowledge_base import (
    KnowledgeBaseRecord,
    load_knowledge_base,
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "trusted_job_scam_guidance"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_PATH = PROJECT_ROOT / "backend" / "data" / "chroma_db"

_model = None
_client = None
_collection = None


def _get_model() -> SentenceTransformer:
    """Load the embedding model once per process."""
    global _model

    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)

    return _model


def _get_collection():
    """Open or create the persistent ChromaDB collection."""
    global _client, _collection

    if _collection is None:
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)

        _client = chromadb.PersistentClient(
            path=str(CHROMA_PATH)
        )

        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    return _collection


def _record_text(record: KnowledgeBaseRecord) -> str:
    """Build searchable text from a trusted guidance record."""
    return " ".join(
        part
        for part in (
            record.title,
            record.publisher,
            record.excerpt,
            " ".join(record.warning_sign_tags),
        )
        if part
    )


def index_guidance(
    records: list[KnowledgeBaseRecord],
) -> int:
    """Embed and index trusted guidance records in ChromaDB."""
    if not records:
        return 0

    collection = _get_collection()
    model = _get_model()

    texts = [_record_text(record) for record in records]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
    ).tolist()

    collection.upsert(
        ids=[record.source_id for record in records],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {
                "publisher": record.publisher,
                "title": record.title,
                "url": record.url,
                "retrieved_date": str(record.retrieved_date),
                "license_note": record.license_note,
                "excerpt": record.excerpt,
                "warning_sign_tags": ",".join(
                    record.warning_sign_tags
                ),
            }
            for record in records
        ],
    )

    return len(records)


def retrieve_semantic(
    query: str,
    *,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Retrieve relevant guidance.

    Similarity is a retrieval score, not a fraud probability or proof.
    """
    if not query or not query.strip():
        return []

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    collection = _get_collection()
    count = collection.count()

    if count == 0:
        return []

    query_embedding = _get_model().encode(
        [query.strip()],
        normalize_embeddings=True,
    ).tolist()

    result = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, count),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    matches = []

    ids = result.get("ids", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    for index, source_id in enumerate(ids):
        metadata = metadatas[index]
        distance = distances[index]

        matches.append(
            {
                "id": source_id,
                "title": metadata["title"],
                "publisher": metadata["publisher"],
                "url": metadata["url"],
                "excerpt": metadata["excerpt"],
                "warning_sign_tags": [
                    tag
                    for tag in metadata.get(
                        "warning_sign_tags", ""
                    ).split(",")
                    if tag
                ],
                "similarity": max(
                    0.0,
                    min(1.0, 1.0 - distance),
                ),
            }
        )

    return matches


def rebuild_guidance_index(
    path: str | Path | None = None,
) -> int:
    """Load and index records from the trusted guidance JSONL file."""
    if path is None:
        path = (
            PROJECT_ROOT
            / "backend"
            / "data"
            / "trusted_guidance_candidates.jsonl"
        )

    records = load_knowledge_base(path)

    return index_guidance(records)