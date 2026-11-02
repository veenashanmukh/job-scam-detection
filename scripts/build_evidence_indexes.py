"""Build the train-only evidence caches used by /api/analyze-job.

Writes backend/models_v2/posting_index.pkl and backend/models_v2/entity_graph.pkl (gitignored),
each storing the train.csv SHA-256 that the app verifies on load. Reads train.csv and only the
job_id column of val.csv; never opens test.csv.

Usage: python scripts/build_evidence_indexes.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services import entity_graph, posting_index  # noqa: E402


def main() -> int:
    processed = ROOT / "data" / "processed_v2"

    start = time.time()
    index = posting_index.build_and_save(processed)
    print(f"posting index: {index.matrix.shape[0]} train postings, {index.matrix.shape[1]} terms "
          f"({time.time() - start:.0f}s) -> {posting_index.DEFAULT_CACHE_PATH}")

    start = time.time()
    graph = entity_graph.build_or_load(processed)
    print(f"entity graph: {graph.graph['n_train']} train postings, {graph.number_of_nodes()} nodes "
          f"({time.time() - start:.0f}s) -> {entity_graph.DEFAULT_CACHE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
