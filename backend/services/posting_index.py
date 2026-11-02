"""Train-only TF-IDF index of job postings, used as similar-posting evidence.

Lexical cosine similarity over the same TF-IDF settings as the baseline. Built from the
TRAIN split only and cached with the train.csv SHA-256, like the entity graph. Snippets
come from the description only (the company profile is left out), so no company
name is shown next to a fraud label unless the description itself names one.
"""
from __future__ import annotations

import json
import pickle
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from backend.services import dataset as ds

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_PATH = PROJECT_ROOT / "backend" / "models_v2" / "posting_index.pkl"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_v2"

SNIPPET_CHARS = 200
CACHE_VERSION = 1


@dataclass
class PostingIndex:
    vectorizer: TfidfVectorizer
    matrix: sparse.csr_matrix
    job_ids: np.ndarray
    labels: np.ndarray
    snippets: list[str]


def make_snippet(description) -> str:
    """Cleaned description, cut to SNIPPET_CHARS at a word boundary where possible."""
    text = ds.clean_field(description) or ""
    if len(text) <= SNIPPET_CHARS:
        return text
    cut = text[: SNIPPET_CHARS - 1]
    space = cut.rfind(" ")
    return (cut[:space] if space > SNIPPET_CHARS // 2 else cut).rstrip() + "…"


def build_index(train_df: pd.DataFrame, *, excluded_job_ids: Collection) -> PostingIndex:
    """Fit on train postings only; any job_id in excluded_job_ids raises ValueError."""
    for column in (ds.ID_COL, ds.LABEL_COL, "text"):
        if column not in train_df.columns:
            raise ValueError(f"train_df is missing column {column!r}")

    excluded = {int(job_id) for job_id in excluded_job_ids}
    overlap = excluded.intersection(int(job_id) for job_id in train_df[ds.ID_COL])
    if overlap:
        raise ValueError(
            f"train_df contains {len(overlap)} excluded job_id(s); the index must be built from train only"
        )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=300_000)
    matrix = vectorizer.fit_transform(train_df["text"].fillna("").astype(str)).tocsr()
    vectorizer.stop_words_ = None  # only needed for introspection; large when pickled

    descriptions = train_df["description"] if "description" in train_df.columns else [None] * len(train_df)
    return PostingIndex(
        vectorizer=vectorizer,
        matrix=matrix,
        job_ids=train_df[ds.ID_COL].to_numpy(dtype=int),
        labels=train_df[ds.LABEL_COL].to_numpy(dtype=int),
        snippets=[make_snippet(d) for d in descriptions],
    )


def search(index: PostingIndex, text: str, top_k: int = 5) -> list[dict]:
    """Top-k train postings by cosine similarity; similarity is relevance, not probability."""
    if not isinstance(text, str) or not text.strip() or top_k < 1:
        return []
    scores = (index.vectorizer.transform([text]) @ index.matrix.T).toarray().ravel()
    if not scores.any():
        return []
    top = np.argsort(-scores, kind="stable")[:top_k]
    return [
        {
            "job_id": int(index.job_ids[i]),
            "label": int(index.labels[i]),
            "similarity": round(float(scores[i]), 4),
            "snippet": index.snippets[i],
        }
        for i in top
        if scores[i] > 0
    ]


def save_index(index: PostingIndex, train_sha256: str, path: str | Path = DEFAULT_CACHE_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump({"version": CACHE_VERSION, "train_sha256": train_sha256, "index": index}, f)


def load_index(expected_train_sha256: str, path: str | Path = DEFAULT_CACHE_PATH) -> PostingIndex:
    """Load the cached index; raise ValueError unless it was built from the expected train.csv."""
    # Local, gitignored cache written by save_index; pickle is not safe for untrusted files.
    with Path(path).open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or payload.get("version") != CACHE_VERSION:
        raise ValueError("posting index cache has an unknown format; rebuild it")
    if payload.get("train_sha256") != expected_train_sha256:
        raise ValueError("posting index cache was built from a different train.csv; rebuild it")
    return payload["index"]


def manifest_train_sha256(processed_dir: str | Path = DEFAULT_PROCESSED_DIR) -> str:
    """Expected train.csv hash, read from the split manifest (no data file is opened)."""
    manifest = json.loads((Path(processed_dir) / "split_manifest.json").read_text(encoding="utf-8"))
    return manifest["splits"]["train"]["sha256"]


def build_and_save(processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
                   cache_path: str | Path = DEFAULT_CACHE_PATH) -> PostingIndex:
    """Build from train.csv (hash-checked against the manifest), excluding val ids; test.csv is never opened."""
    processed_dir = Path(processed_dir)
    manifest = json.loads((processed_dir / "split_manifest.json").read_text(encoding="utf-8"))
    train_path = processed_dir / manifest["splits"]["train"]["file"]
    train_sha = ds.file_sha256(train_path)
    if train_sha != manifest["splits"]["train"]["sha256"]:
        raise ValueError("train.csv does not match the split manifest; refusing to build the index")

    val_ids = pd.read_csv(processed_dir / manifest["splits"]["val"]["file"], usecols=[ds.ID_COL])[ds.ID_COL]
    index = build_index(pd.read_csv(train_path), excluded_job_ids=val_ids)
    save_index(index, train_sha, cache_path)
    return index
