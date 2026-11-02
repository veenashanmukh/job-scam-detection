"""Recruiter/company entity graph built from the TRAIN split only.

EMSCAD already replaces URLs, emails and phone numbers with hashed tokens such as
``#EMAIL_<hex>#``. Those hashes, plus a hashed company key taken from the first line of
the company profile, become entity nodes linked to the postings that mention them.
The graph is used for explanation ("this email also appears in N known fraud postings")
and as features for an exploratory validation-only combiner. Nothing here returns raw
contact details or profile text.
"""
from __future__ import annotations

import hashlib
import html
import json
import pickle
import re
from collections.abc import Collection, Mapping
from pathlib import Path

import networkx as nx
import pandas as pd

from backend.services import dataset as ds

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_PATH = PROJECT_ROOT / "backend" / "models_v2" / "entity_graph.pkl"

TOKEN_RE = re.compile(r"#(URL|EMAIL|PHONE)_([0-9a-f]{16,})#")
ENTITY_FIELDS = ("company_profile", "description", "requirements", "benefits")
ENTITY_KINDS = ("EMAIL", "PHONE", "URL", "COMPANY")  # also the priority for shared_entity

HUB_MAX_POSTINGS = 50   # entities linked to more postings than this are dropped
FRAUD_PRIOR = 0.0475    # train fraud rate (594 / 12,504)
PRIOR_WEIGHT = 2.0      # pseudo-postings behind the prior in fraud_rate_smoothed
MAX_LINKED_RETURNED = 20
CACHE_VERSION = 1

_BLOCK_TAG = re.compile(r"<\s*(?:br|/p|/h\d|/li|/div|/ul|/ol)\s*/?>", re.IGNORECASE)
_ANY_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_EDGE_PUNCT = " .,:;!?-–—\"'()[]"


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def normalise_company(profile) -> str | None:
    """First non-empty line of the profile, HTML removed, lowercased, edges stripped."""
    if not isinstance(profile, str) or not profile.strip():
        return None
    text = _ANY_TAG.sub(" ", _BLOCK_TAG.sub("\n", html.unescape(profile)))
    for line in text.splitlines():
        line = _WS.sub(" ", TOKEN_RE.sub(" ", line)).strip().lower().strip(_EDGE_PUNCT)
        if line:
            return line[:120]
    return None


def extract_entities(posting_fields: Mapping) -> dict:
    """
    Return {"url": [...], "email": [...], "phone": [...], "company": str | None}.

    Token lists hold the dataset's existing hashes (sorted, de-duplicated) found in
    company_profile, description, requirements and benefits. company is the normalised
    first line of company_profile, or None when the profile is missing or empty.
    """
    found = {"URL": set(), "EMAIL": set(), "PHONE": set()}
    for field in ENTITY_FIELDS:
        value = posting_fields.get(field)
        if isinstance(value, str):
            for kind, digest in TOKEN_RE.findall(value):
                found[kind].add(digest)

    return {
        "url": sorted(found["URL"]),
        "email": sorted(found["EMAIL"]),
        "phone": sorted(found["PHONE"]),
        "company": normalise_company(posting_fields.get("company_profile")),
    }


def entity_ids(entities: dict) -> list[str]:
    """Graph node ids "KIND:<hash>"; the company key is hashed so no text is stored."""
    ids = [f"EMAIL:{h}" for h in entities["email"]]
    ids += [f"PHONE:{h}" for h in entities["phone"]]
    ids += [f"URL:{h}" for h in entities["url"]]
    if entities["company"]:
        ids.append(f"COMPANY:{_short_hash(entities['company'])}")
    return ids


def build_graph(train_df: pd.DataFrame, *, excluded_job_ids: Collection) -> nx.Graph:
    """
    Bipartite graph of train postings and their entities.

    excluded_job_ids must hold every job_id that may not be used (validation, test);
    any overlap with train_df raises ValueError.
    """
    for column in (ds.ID_COL, ds.LABEL_COL):
        if column not in train_df.columns:
            raise ValueError(f"train_df is missing column {column!r}")

    excluded = {int(job_id) for job_id in excluded_job_ids}
    overlap = excluded.intersection(int(job_id) for job_id in train_df[ds.ID_COL])
    if overlap:
        raise ValueError(
            f"train_df contains {len(overlap)} excluded job_id(s); the graph must be built from train only"
        )

    graph = nx.Graph()
    for row in train_df.to_dict("records"):
        posting = ("job", int(row[ds.ID_COL]))
        graph.add_node(posting, bipartite=0, label=int(row[ds.LABEL_COL]))
        for entity in entity_ids(extract_entities(row)):
            graph.add_node(entity, bipartite=1, kind=entity.split(":", 1)[0])
            graph.add_edge(posting, entity)

    hubs = [n for n, d in graph.nodes(data=True) if d["bipartite"] == 1 and graph.degree(n) > HUB_MAX_POSTINGS]
    graph.remove_nodes_from(hubs)
    graph.graph.update(hubs_removed=len(hubs), hub_max_postings=HUB_MAX_POSTINGS, n_train=len(train_df))
    return graph


def _smoothed_rate(fraud: int, total: int) -> float:
    return (fraud + FRAUD_PRIOR * PRIOR_WEIGHT) / (total + PRIOR_WEIGHT)


def _linked(graph: nx.Graph, posting_fields: Mapping, exclude_job_id=None, kinds=ENTITY_KINDS):
    """(entity ids present in the graph, {job_id: (label, first shared entity)})."""
    present = [e for e in entity_ids(extract_entities(posting_fields))
               if e in graph and e.split(":", 1)[0] in kinds]
    present.sort(key=lambda e: ENTITY_KINDS.index(e.split(":", 1)[0]))
    linked = {}
    for entity in present:
        for _, job_id in graph.neighbors(entity):
            if job_id != exclude_job_id and job_id not in linked:
                linked[job_id] = (graph.nodes[("job", job_id)]["label"], entity)
    return present, linked


def balanced_examples(postings, limit: int = MAX_LINKED_RETURNED) -> list[tuple[int, int]]:
    """
    Up to `limit` (job_id, label) pairs for display: up to half fraud and half legitimate when both exist, the other
    side topping up if one is short, picked by job_id so the choice is stable. No fraud-first bias.
    """
    fraud = sorted(job_id for job_id, label in postings if label == 1)
    legit = sorted(job_id for job_id, label in postings if label == 0)
    n_fraud = min(len(fraud), limit // 2)
    n_legit = min(len(legit), limit - n_fraud)
    n_fraud = min(len(fraud), limit - n_legit)
    return sorted([(job_id, 1) for job_id in fraud[:n_fraud]] + [(job_id, 0) for job_id in legit[:n_legit]])


def entity_links(graph: nx.Graph, posting_fields: Mapping, exclude_job_id=None) -> list[dict]:
    """
    One entry per matched entity (email, phone, URL, company profile) with ITS OWN true totals: the number of train
    postings linked to that entity alone and how many are labelled fraud. Uncapped and independent of the other
    entities, so a posting shared by several entities counts under each. Only the example list is limited.
    """
    present = [e for e in entity_ids(extract_entities(posting_fields)) if e in graph]
    present.sort(key=lambda e: ENTITY_KINDS.index(e.split(":", 1)[0]))
    links = []
    for entity in present:
        postings = [(job_id, graph.nodes[("job", job_id)]["label"]) for _, job_id in graph.neighbors(entity)
                    if job_id != exclude_job_id]
        if not postings:
            continue
        links.append({
            "entity": entity,
            "kind": entity.split(":", 1)[0],
            "linked_total": len(postings),
            "fraud_total": sum(label for _, label in postings),
            "examples": [{"job_id": job_id, "label": label} for job_id, label in balanced_examples(postings)],
        })
    return links


def query(graph: nx.Graph, posting_fields: Mapping, exclude_job_id=None) -> dict:
    """Explanation-ready summary of train postings that share an entity with this posting."""
    present, linked = _linked(graph, posting_fields, exclude_job_id)
    fraud = sum(label for label, _ in linked.values())
    chosen = balanced_examples([(job_id, label) for job_id, (label, _) in linked.items()])

    return {
        "entities_found": present,
        "entities": entity_links(graph, posting_fields, exclude_job_id),
        "linked_postings": [
            {"job_id": job_id, "label": label, "shared_entity": linked[job_id][1]} for job_id, label in chosen
        ],
        "linked_postings_total": len(linked),
        "fraud_neighbour_count": fraud,
        "fraud_rate_smoothed": _smoothed_rate(fraud, len(linked)),
    }


def graph_features(graph: nx.Graph, posting_fields: Mapping, exclude_job_id=None) -> dict:
    """Numeric features for the combiner: all entities, and email/phone only."""
    _, linked = _linked(graph, posting_fields, exclude_job_id)
    _, contact = _linked(graph, posting_fields, exclude_job_id, kinds=("EMAIL", "PHONE"))
    fraud = sum(label for label, _ in linked.values())
    return {
        "ents_n": len(linked),
        "ents_fraud": fraud,
        "ents_rate": _smoothed_rate(fraud, len(linked)),
        "emailphone_n": len(contact),
        "emailphone_fraud": sum(label for label, _ in contact.values()),
    }


def save_graph(graph: nx.Graph, train_sha256: str, path: str | Path = DEFAULT_CACHE_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump({"version": CACHE_VERSION, "train_sha256": train_sha256, "graph": graph}, f)


def load_graph(expected_train_sha256: str, path: str | Path = DEFAULT_CACHE_PATH) -> nx.Graph:
    """Load a cached graph; raise ValueError unless it was built from the expected train.csv."""
    # Local, gitignored cache written by save_graph; pickle is not safe for untrusted files.
    with Path(path).open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or payload.get("version") != CACHE_VERSION:
        raise ValueError("entity graph cache has an unknown format; rebuild it")
    if payload.get("train_sha256") != expected_train_sha256:
        raise ValueError("entity graph cache was built from a different train.csv; rebuild it")
    return payload["graph"]


def build_or_load(processed_dir: str | Path, cache_path: str | Path = DEFAULT_CACHE_PATH) -> nx.Graph:
    """
    Load the cached graph or build it from train.csv.

    train.csv must match the split manifest's SHA-256 (the pipeline guarantees that split
    shares no job_id or group with val/test). Validation ids are excluded explicitly;
    test.csv is never opened.
    """
    processed_dir = Path(processed_dir)
    manifest = json.loads((processed_dir / "split_manifest.json").read_text(encoding="utf-8"))
    train_path = processed_dir / manifest["splits"]["train"]["file"]
    train_sha = ds.file_sha256(train_path)
    if train_sha != manifest["splits"]["train"]["sha256"]:
        raise ValueError("train.csv does not match the split manifest; refusing to build the graph")

    cache_path = Path(cache_path)
    if cache_path.exists():
        try:
            return load_graph(train_sha, cache_path)
        except ValueError:
            pass  # stale cache: rebuild below

    val_ids = pd.read_csv(processed_dir / manifest["splits"]["val"]["file"], usecols=[ds.ID_COL])[ds.ID_COL]
    graph = build_graph(pd.read_csv(train_path), excluded_job_ids=val_ids)
    save_graph(graph, train_sha, cache_path)
    return graph
