"""Synthetic fixture for unit tests ONLY.

This is NOT EMSCAD and says nothing about it. It mimics the column layout so the pipeline logic can be tested,
including planted exact duplicates, near-duplicates, empty rows and a label-conflict duplicate.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TOPICS = ["warehouse", "nursing", "software", "retail", "driver", "teacher", "accounting", "welding", "design", "support"]
WORDS = ("team quality customer process project system report plan market service budget client skill training "
         "safety review manager develop deliver maintain support improve analyse coordinate schedule document").split()


def _desc(rng, topic: str, n: int = 60) -> str:
    # topic keyword is repeated and a per-row random bag of words makes distinct rows clearly different
    return f"We seek a {topic} specialist. " + " ".join(rng.choice(WORDS, size=n)) + f" {topic} " + " ".join(rng.choice(WORDS, size=20))


def make_fixture(n=400, fraud_rate=0.06, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        topic = TOPICS[i % len(TOPICS)]
        unique = f"ref{i:05d}x{rng.integers(1, 10**6)}"
        rows.append({
            "job_id": 1000 + i,
            "title": f"{topic.title()} position {unique}",
            "location": "US, NY, New York",
            "department": None,
            "salary_range": None,
            "company_profile": f"Company {unique} " + " ".join(rng.choice(WORDS, size=15)),
            "description": f"<p>{_desc(rng, topic)} {unique}</p>",
            "requirements": " ".join(rng.choice(WORDS, size=12)) + f" {unique}",
            "benefits": None if i % 3 == 0 else " ".join(rng.choice(WORDS, size=8)) + f" {unique}",
            "telecommuting": 0, "has_company_logo": 1, "has_questions": 0,
            "employment_type": "Full-time", "required_experience": None, "required_education": None,
            "industry": None, "function": None,
            "fraudulent": int(rng.random() < fraud_rate),
        })
    df = pd.DataFrame(rows)
    # make sure there are enough fraud rows regardless of seed
    df.loc[df.index[:: n // 24][:24], "fraudulent"] = 1

    extra = []
    nxt = 5000
    # exact duplicates (same text, new ids), 3 groups of 2-3 copies; fraud postings among them
    for src in (5, 11, 17):
        for _ in range(2):
            r = df.iloc[src].copy()
            r["job_id"] = nxt
            nxt += 1
            extra.append(r)
    # near duplicate: tiny edit of one posting (one extra word appended)
    r = df.iloc[23].copy()
    r["job_id"] = nxt
    r["description"] = r["description"] + " apply today"
    nxt += 1
    extra.append(r)
    # label-conflict duplicate: same text, opposite label
    r = df.iloc[29].copy()
    r["job_id"] = nxt
    r["fraudulent"] = 1 - int(df.iloc[29]["fraudulent"])
    nxt += 1
    extra.append(r)
    # two rows with every text field empty (must NOT be grouped together)
    for _ in range(2):
        r = df.iloc[40].copy()
        for f in ("title", "company_profile", "description", "requirements", "benefits"):
            r[f] = None if f != "title" else "  "
        r["job_id"] = nxt
        r["fraudulent"] = 0
        nxt += 1
        extra.append(r)
    return pd.concat([df, pd.DataFrame(extra)], ignore_index=True)


@pytest.fixture(scope="session")
def fixture_df() -> pd.DataFrame:
    return make_fixture()


@pytest.fixture()
def fixture_csv(tmp_path, fixture_df) -> Path:
    p = tmp_path / "fake_job_postings.csv"
    fixture_df.to_csv(p, index=False)
    return p
