"""EMSCAD dataset pipeline: validation, profiling, duplicate grouping, shared split, leakage checks.

Design rules (see reports/phase1_notes.md):
  * Nothing here invents content. Missing fields stay missing; the model text only contains fields that exist.
  * The split is group-aware: exact and near-duplicate postings share a group_id and never cross split boundaries.
  * The split is deterministic (seed 42) and recorded in a manifest with file hashes.
  * The test split can only be loaded with final_evaluation=True.
"""
from __future__ import annotations

import hashlib
import html
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

SEED = 42
ID_COL = "job_id"
LABEL_COL = "fraudulent"
GROUP_COL = "group_id"
SPLIT_COL = "split"
TEXT_FIELDS = ["title", "company_profile", "description", "requirements", "benefits"]
METADATA_FIELDS = [
    "location", "department", "salary_range", "telecommuting", "has_company_logo", "has_questions",
    "employment_type", "required_experience", "required_education", "industry", "function",
]
REQUIRED_COLUMNS = [ID_COL, LABEL_COL, *TEXT_FIELDS]
SPLIT_NAMES = ("train", "val", "test")
DEFAULT_FRACTIONS = (0.70, 0.15, 0.15)
DEFAULT_NEAR_THRESHOLD = 0.90
# Float tolerance applied identically in grouping and in the leakage check, so a pair sitting on the threshold
# cannot be grouped by one and missed by the other because of last-bit rounding. Conservative (treats it as duplicate).
NEAR_EPS = 1e-9

_HTML_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
_PHONE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?){2,4}\d{2,4}(?!\d)")
_PLACEHOLDER = re.compile(r"#(?:URL|EMAIL|PHONE)_[0-9A-Za-z]+#")


class SchemaError(ValueError):
    """Raised when the raw file does not match the expected schema."""


# --------------------------------------------------------------------------- loading / validation



_LABEL_MAP = {
    "f": 0, "false": 0, "0": 0, "0.0": 0,
    "t": 1, "true": 1, "1": 1, "1.0": 1,
}


def normalize_labels(values: pd.Series) -> pd.Series:
    """Convert supported fraud labels to int8 0/1."""
    as_text = values.astype("string").str.strip().str.lower()
    mapped = as_text.map(_LABEL_MAP)
    bad = mapped.isna()

    if bad.any():
        unknown = sorted(
            as_text[bad].fillna("<missing>").unique().tolist()
        )
        raise SchemaError(
            f"Unrecognised '{values.name}' labels in {int(bad.sum())} rows: "
            f"{unknown[:10]}. Expected one of {sorted(_LABEL_MAP)}."
        )

    return mapped.astype("int8")


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the raw CSV, generate internal IDs if absent, and normalize labels."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. "
                    "Download fake_job_postings.csv (EMSCAD, Kaggle mirror) and place it in data/raw/; see README 'Dataset'."
        )

    df = pd.read_csv(path)

    # EMSCAD has no native job_id; IDs follow the current CSV row order.
    if ID_COL not in df.columns:
        df.insert(0, ID_COL, range(1, len(df) + 1))

    if LABEL_COL in df.columns:
        df[LABEL_COL] = normalize_labels(df[LABEL_COL])

    return df



def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_schema(df: pd.DataFrame) -> dict:
    """Hard failures raise SchemaError; soft issues are returned as warnings."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {missing}")
    if df[ID_COL].isna().any():
        raise SchemaError("job_id contains missing values")
    if df[ID_COL].duplicated().any():
        raise SchemaError(f"job_id is not unique ({int(df[ID_COL].duplicated().sum())} repeated ids)")
    if df[LABEL_COL].isna().any():
        raise SchemaError("fraudulent contains missing values")
    labels = set(pd.unique(df[LABEL_COL]))
    if not labels <= {0, 1}:
        raise SchemaError(f"fraudulent must be binary 0/1, found {sorted(labels, key=str)}")
    warnings = []
    absent_meta = [c for c in METADATA_FIELDS if c not in df.columns]
    if absent_meta:
        warnings.append(f"Metadata columns absent (metadata for those fields is unavailable): {absent_meta}")
    known = set(REQUIRED_COLUMNS) | set(METADATA_FIELDS)
    extra = [c for c in df.columns if c not in known]
    if extra:
        warnings.append(f"Unexpected extra columns kept as-is: {extra}")
    return {"n_rows": int(len(df)), "n_cols": int(df.shape[1]), "warnings": warnings}


# --------------------------------------------------------------------------- text construction
def clean_field(value) -> str | None:
    """Unescape HTML entities, strip tags, collapse whitespace. Empty or missing -> None (never filled in)."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    s = html.unescape(str(value))
    s = _HTML_TAG.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s or None


_SECTION_LABELS = {
    "title": "Title",
    "company_profile": "Company profile",
    "description": "Description",
    "requirements": "Requirements",
    "benefits": "Benefits",
}


def build_text(row: pd.Series) -> str:
    parts = []
    for field in TEXT_FIELDS:
        cleaned = clean_field(row.get(field))
        if cleaned:
            parts.append(f"{_SECTION_LABELS[field]}: {cleaned}")
    return "\n".join(parts)


def add_model_text(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["text"] = out.apply(build_text, axis=1)
    out["n_text_fields_present"] = out[TEXT_FIELDS].apply(lambda r: sum(clean_field(v) is not None for v in r), axis=1)
    return out


# --------------------------------------------------------------------------- profiling
def _length_stats(s: pd.Series) -> dict:
    if len(s) == 0:
        return {}
    q = s.quantile([0.5, 0.9, 0.99])
    return {"mean": float(s.mean()), "median": float(q[0.5]), "p90": float(q[0.9]), "p99": float(q[0.99]), "max": int(s.max())}


def profile(df: pd.DataFrame) -> dict:
    """df must already contain 'text' (call add_model_text first)."""
    n = len(df)
    counts = df[LABEL_COL].value_counts().sort_index()
    n_fraud = int(counts.get(1, 0))
    prof: dict = {
        "n_rows": int(n),
        "label_counts": {int(k): int(v) for k, v in counts.items()},
        "fraud_rate": float(n_fraud / n) if n else None,
    }
    present_cols = [c for c in [*TEXT_FIELDS, *METADATA_FIELDS] if c in df.columns]
    miss = {}
    for c in present_cols:
        is_missing = df[c].map(lambda v: clean_field(v) is None) if c in TEXT_FIELDS else df[c].isna()
        miss[c] = {
            "missing": int(is_missing.sum()),
            "missing_pct": float(is_missing.mean() * 100),
            "missing_pct_in_fraud": float(is_missing[df[LABEL_COL] == 1].mean() * 100) if n_fraud else None,
            "missing_pct_in_legit": float(is_missing[df[LABEL_COL] == 0].mean() * 100) if n - n_fraud else None,
        }
    prof["missingness"] = miss

    words = df["text"].str.split().map(len)
    chars = df["text"].str.len()
    prof["text_length_words"] = {
        "all": _length_stats(words),
        "fraud": _length_stats(words[df[LABEL_COL] == 1]),
        "legit": _length_stats(words[df[LABEL_COL] == 0]),
    }
    prof["text_length_chars"] = _length_stats(chars)
    # Words are not tokens; ~200 words is only a rough marker for a 256-token limit. Measure real tokens in Phase 2.
    prof["pct_over_200_words"] = float((words > 200).mean() * 100)
    prof["empty_text_rows"] = int((df["text"].str.len() == 0).sum())

    raw_text_all = df[TEXT_FIELDS].fillna("").astype(str).agg(" ".join, axis=1)
    prof["html_tag_rows"] = int(raw_text_all.str.contains(_HTML_TAG).sum())
    prof["contact_patterns"] = {
        "rows_with_email_like": int(raw_text_all.str.contains(_EMAIL).sum()),
        "rows_with_url_like": int(raw_text_all.str.contains(_URL).sum()),
        "rows_with_phone_like": int(raw_text_all.str.contains(_PHONE).sum()),
        "rows_with_placeholder_tokens": int(raw_text_all.str.contains(_PLACEHOLDER).sum()),
        "placeholder_token_total": int(raw_text_all.str.count(_PLACEHOLDER).sum()),
        "note": "Pattern-based counts; phone regex is loose and over-counts digit sequences. Placeholder tokens "
                "(#URL_xxx#, #EMAIL_xxx#, #PHONE_xxx#) indicate redaction. Confirm by manual inspection.",
    }
    return prof


# --------------------------------------------------------------------------- duplicates
class _UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return int(x)

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def normalize_for_dedupe(text: str) -> str:
    t = _PLACEHOLDER.sub(" ", text.lower())
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return _WS.sub(" ", t).strip()


def _tfidf(norm_texts: pd.Series | list[str]):
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, dtype=np.float64)
    return vec.fit_transform(norm_texts)


def near_duplicate_pairs(norm_texts: list[str], threshold: float = DEFAULT_NEAR_THRESHOLD,
                         chunk: int = 500) -> list[tuple[int, int, float]]:
    """All (i, j, cosine), i < j, with cosine >= threshold on ONE TF-IDF fit over `norm_texts`.
    This is the single similarity definition shared by grouping and the leakage check."""
    if len(norm_texts) < 2:
        return []
    X = _tfidf(norm_texts)
    out: list[tuple[int, int, float]] = []
    for start in range(0, len(norm_texts), chunk):
        S = (X[start:start + chunk] @ X.T).tocoo()
        rows = S.row + start
        keep = (S.data >= threshold - NEAR_EPS) & (rows < S.col)
        out.extend((int(r), int(c), float(v)) for r, c, v in zip(rows[keep], S.col[keep], S.data[keep]))
    return out


def assign_duplicate_groups(
    df: pd.DataFrame, near_threshold: float = DEFAULT_NEAR_THRESHOLD, chunk: int = 500
) -> tuple[pd.Series, dict]:
    """Group exact and near-duplicate postings (cosine >= near_threshold on word 1-2gram TF-IDF).

    Rows with empty normalised text are never grouped together. Returns (group_id Series aligned to df.index,
    stats). group_id is the smallest job_id in the group, so it is stable.
    """
    n = len(df)
    ids = df[ID_COL].to_numpy()
    norm = df["text"].map(normalize_for_dedupe).reset_index(drop=True)
    uf = _UnionFind(n)

    first_seen: dict[str, int] = {}
    n_exact_extra = 0
    for i, t in enumerate(norm):
        if not t:
            continue
        key = hashlib.sha1(t.encode("utf-8")).hexdigest()
        if key in first_seen:
            uf.union(first_seen[key], i)
            n_exact_extra += 1
        else:
            first_seen[key] = i

    nonempty = np.flatnonzero((norm.str.len() > 0).to_numpy())
    for r, c, _ in near_duplicate_pairs(norm.iloc[nonempty].tolist(), near_threshold, chunk):
        uf.union(int(nonempty[r]), int(nonempty[c]))

    roots = np.array([uf.find(i) for i in range(n)])
    # group id = smallest job_id within each root's members
    group_min = pd.Series(ids).groupby(roots).transform("min").to_numpy()
    group = pd.Series(group_min, index=df.index, name=GROUP_COL)

    sizes = group.value_counts()
    labels = df.groupby(group)[LABEL_COL].nunique()
    stats = {
        "near_threshold": near_threshold,
        "n_rows": int(n),
        "n_groups": int(sizes.size),
        "rows_in_multi_member_groups": int(sizes[sizes > 1].sum()),
        "n_multi_member_groups": int((sizes > 1).sum()),
        "exact_duplicate_rows_beyond_first": int(n_exact_extra),
        "largest_group_size": int(sizes.max()) if len(sizes) else 0,
        "largest_group_pct_of_rows": float(sizes.max() / n * 100) if n else 0.0,
        "group_size_distribution": {int(k): int(v) for k, v in sizes.value_counts().sort_index().items()},
        "groups_with_conflicting_labels": int((labels > 1).sum()),
    }
    return group, stats


# --------------------------------------------------------------------------- split
def group_stratified_split(
    df: pd.DataFrame,
    fractions: tuple[float, float, float] = DEFAULT_FRACTIONS,
    seed: int = SEED,
) -> pd.Series:
    """Assign each group to train/val/test. Stratified on group label (a group is 'fraud' if any member is).

    Groups are shuffled within each stratum (seeded), then each is assigned to the split with the largest remaining
    relative deficit. Deterministic for a given seed and input order of group ids.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1")
    g = df.groupby(GROUP_COL).agg(n=(LABEL_COL, "size"), label=(LABEL_COL, "max")).sort_index()
    rng = np.random.default_rng(seed)
    assignment: dict[int, str] = {}
    for lab in sorted(g["label"].unique()):
        sub = g[g["label"] == lab]
        sub = sub.iloc[rng.permutation(len(sub))]
        total = int(sub["n"].sum())
        targets = np.array(fractions, dtype=float) * total
        counts = np.zeros(3)
        for gid, n in zip(sub.index, sub["n"]):
            deficit = (targets - counts) / np.maximum(targets, 1e-9)
            k = int(np.argmax(deficit))
            assignment[gid] = SPLIT_NAMES[k]
            counts[k] += n
    return df[GROUP_COL].map(assignment).rename(SPLIT_COL)


# --------------------------------------------------------------------------- leakage checks
def max_cross_similarity(a_texts: list[str], b_texts: list[str], chunk: int = 500) -> np.ndarray:
    """LEGACY, split-dependent: max cosine of each A text to B with TF-IDF fit on A+B only. The value for a given pair
    changes with which other rows are in A and B, so it is NOT used for gating any more (see check_leakage); kept for
    diagnostics."""
    if not a_texts or not b_texts:
        return np.zeros(len(a_texts))
    X = _tfidf(list(a_texts) + list(b_texts))
    XA, XB = X[: len(a_texts)], X[len(a_texts):]
    out = np.zeros(len(a_texts))
    for start in range(0, len(a_texts), chunk):
        S = XA[start:start + chunk] @ XB.T
        out[start:start + chunk] = S.max(axis=1).toarray().ravel()
    return out


def _corpus_split_matrices(splits: dict[str, pd.DataFrame]) -> dict[str, tuple]:
    """One TF-IDF fit over every non-empty normalised text in all splits; returns {name: (matrix_rows, positions)}.
    Because the fit is over the whole corpus, a pair's similarity does not depend on how rows are split."""
    names = list(splits)
    norm = {k: splits[k]["text"].map(normalize_for_dedupe).tolist() for k in names}
    pos = {k: [i for i, t in enumerate(norm[k]) if t] for k in names}
    texts = [norm[k][i] for k in names for i in pos[k]]
    if not texts:
        return {}
    X = _tfidf(texts)
    out, off = {}, 0
    for k in names:
        out[k] = (X[off:off + len(pos[k])], pos[k])
        off += len(pos[k])
    return out


def check_leakage(splits: dict[str, pd.DataFrame], near_threshold: float = DEFAULT_NEAR_THRESHOLD) -> dict:
    """Run all cross-split checks. Each check reports pass/fail plus details."""
    names = list(splits)
    report: dict = {"checks": {}, "near_threshold": near_threshold}

    def _pairs():
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                yield names[i], names[j]

    ids_ok, groups_ok, text_ok = True, True, True
    details = {"job_id_overlap": {}, "group_overlap": {}, "exact_text_overlap": {}}
    norm = {k: set(v["text"].map(normalize_for_dedupe)) - {""} for k, v in splits.items()}
    for a, b in _pairs():
        key = f"{a}|{b}"
        o_id = set(splits[a][ID_COL]) & set(splits[b][ID_COL])
        o_gr = set(splits[a][GROUP_COL]) & set(splits[b][GROUP_COL])
        o_tx = norm[a] & norm[b]
        details["job_id_overlap"][key] = len(o_id)
        details["group_overlap"][key] = len(o_gr)
        details["exact_text_overlap"][key] = len(o_tx)
        ids_ok &= not o_id
        groups_ok &= not o_gr
        text_ok &= not o_tx
    report["checks"]["no_job_id_overlap"] = {"passed": bool(ids_ok), "details": details["job_id_overlap"]}
    report["checks"]["no_group_overlap"] = {"passed": bool(groups_ok), "details": details["group_overlap"]}
    report["checks"]["no_exact_text_overlap"] = {"passed": bool(text_ok), "details": details["exact_text_overlap"]}

    near_ok, near_details = True, {}
    mats = _corpus_split_matrices(splits)
    for later, earlier in (("val", "train"), ("test", "train"), ("test", "val")):
        if later in splits and earlier in splits:
            XA, _ = mats.get(later, (None, []))
            XB, _ = mats.get(earlier, (None, []))
            if XA is None or XB is None or XA.shape[0] == 0 or XB.shape[0] == 0:
                m = np.zeros(0)
            else:
                m = np.zeros(XA.shape[0])
                for start in range(0, XA.shape[0], 500):
                    S = XA[start:start + 500] @ XB.T
                    m[start:start + 500] = S.max(axis=1).toarray().ravel()
            n_hit = int((m >= near_threshold - NEAR_EPS).sum())
            a_norm = [t for t in splits[later]["text"].map(normalize_for_dedupe).tolist() if t]
            b_norm = [t for t in splits[earlier]["text"].map(normalize_for_dedupe).tolist() if t]
            legacy = max_cross_similarity(a_norm, b_norm)
            near_details[f"{later}->{earlier}"] = {
                "max_similarity": float(m.max()) if len(m) else 0.0,
                "p99_similarity": float(np.quantile(m, 0.99)) if len(m) else 0.0,
                "n_rows_at_or_above_threshold": n_hit,
                # Diagnostic only, NOT gating: split-dependent TF-IDF (fit on these two splits alone).
                "legacy_split_fit_rows_at_or_above": int((legacy >= near_threshold).sum()),
            }
            near_ok &= n_hit == 0
    report["checks"]["no_cross_split_near_duplicates"] = {"passed": bool(near_ok), "details": near_details}
    report["all_passed"] = bool(all(c["passed"] for c in report["checks"].values()))
    return report


# --------------------------------------------------------------------------- near-duplicate resolution (NEW)
class GroupTooLargeError(RuntimeError):
    """Merging near-duplicate groups produced a group large enough to distort the split."""


class ResolutionInconsistencyError(RuntimeError):
    """Grouping and leakage detection disagree even though they share one similarity definition (a bug)."""


def find_cross_split_near_pairs(splits: dict[str, pd.DataFrame], near_threshold: float = DEFAULT_NEAR_THRESHOLD,
                                chunk: int = 500) -> list[tuple[int, int, float]]:
    """Every (job_id_later, job_id_earlier, cosine) at/above threshold across the split pairs that check_leakage
    compares, using the SAME corpus-fit similarity as check_leakage. Returns the pairs, not just per-row maxima."""
    mats = _corpus_split_matrices(splits)
    pairs: list[tuple[int, int, float]] = []
    for later, earlier in (("val", "train"), ("test", "train"), ("test", "val")):
        if later not in mats or earlier not in mats:
            continue
        XA, a_pos = mats[later]
        XB, b_pos = mats[earlier]
        if XA.shape[0] == 0 or XB.shape[0] == 0:
            continue
        a_ids, b_ids = splits[later][ID_COL].to_numpy(), splits[earlier][ID_COL].to_numpy()
        for start in range(0, XA.shape[0], chunk):
            S = (XA[start:start + chunk] @ XB.T).tocoo()
            keep = S.data >= near_threshold - NEAR_EPS
            for r, c, v in zip(S.row[keep], S.col[keep], S.data[keep]):
                pairs.append((int(a_ids[a_pos[start + r]]), int(b_ids[b_pos[c]]), float(v)))
    return pairs


def merge_groups_by_pairs(df: pd.DataFrame, pairs: list[tuple[int, int, float]]) -> pd.Series:
    """Union the groups of every pair's two job_ids. New group_id = smallest old group_id in the merged set
    (old ids are smallest job_ids, so the id stays stable and equals the smallest job_id)."""
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    gid_of = df.set_index(ID_COL)[GROUP_COL]
    uniq = pd.Index(df[GROUP_COL].unique())
    pos = {g: i for i, g in enumerate(uniq)}
    ea = [pos[gid_of[a]] for a, _, _ in pairs]
    eb = [pos[gid_of[b]] for _, b, _ in pairs]
    n = len(uniq)
    graph = sparse.coo_matrix((np.ones(len(ea)), (ea, eb)), shape=(n, n))
    comp = connected_components(graph, directed=False)[1]
    new_id = pd.Series(uniq.to_numpy()).groupby(comp).transform("min").to_numpy()
    return df[GROUP_COL].map(dict(zip(uniq, new_id)))


def resolve_cross_split_near_duplicates(
    df: pd.DataFrame,
    fractions: tuple[float, float, float] = DEFAULT_FRACTIONS,
    seed: int = SEED,
    near_threshold: float = DEFAULT_NEAR_THRESHOLD,
    max_group_fraction: float = 0.05,
    grouping_margin: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    """Deterministic, single-pass replacement for the earlier iterative resolver.

    Why the iterative one never converged: the old leakage similarity was fit on the two splits being compared, so a
    pair's cosine moved with every re-split (0.899 in one split, 0.901 in the next). Here grouping and the check
    share ONE corpus-fit definition, so the set of pairs >= threshold is a fixed graph independent of the split:
      1. find ALL such pairs over the whole corpus (not only ones that currently cross splits),
      2. merge their groups (transitively),
      3. split ONCE, then
      4. verify with the independent cross-split finder that no pair crosses; raise otherwise.
    Threshold and final check_leakage gate are unchanged.

    grouping_margin (default 0.0) optionally groups at near_threshold - margin: a STRICTER buffer so pairs just under
    the threshold also stay together. It can only add merges; the gate threshold itself never moves."""
    if not 0.0 <= grouping_margin < near_threshold:
        raise ValueError("grouping_margin must be in [0, near_threshold)")
    group_threshold = near_threshold - grouping_margin
    df = df.copy()
    norm = df["text"].map(normalize_for_dedupe).tolist()
    pos = [i for i, t in enumerate(norm) if t]
    ids = df[ID_COL].to_numpy()
    all_pairs = [(int(ids[pos[a]]), int(ids[pos[b]]), v)
                 for a, b, v in near_duplicate_pairs([norm[i] for i in pos], group_threshold)]

    gid_of = df.set_index(ID_COL)[GROUP_COL]
    merging = [(a, b, v) for a, b, v in all_pairs if gid_of[a] != gid_of[b]]
    n_before = int(df[GROUP_COL].nunique())
    if all_pairs:
        df[GROUP_COL] = merge_groups_by_pairs(df, all_pairs)
    sizes = df[GROUP_COL].value_counts()
    if int(sizes.max()) > max_group_fraction * len(df):
        raise GroupTooLargeError(
            f"After merging near-duplicate pairs the largest group has {int(sizes.max())} rows "
            f"(> {max_group_fraction:.0%} of {len(df)}). Review the grouping text representation; "
            "do not exempt pairs or raise the threshold."
        )

    df[SPLIT_COL] = group_stratified_split(df, fractions=fractions, seed=seed)
    parts = {name: df[df[SPLIT_COL] == name] for name in SPLIT_NAMES}
    crossing = find_cross_split_near_pairs(parts, group_threshold)
    if crossing:
        raise ResolutionInconsistencyError(
            f"{len(crossing)} pair(s) still cross splits after merging all pairs under one similarity definition; "
            f"splits not saved. First few: {crossing[:5]}"
        )
    return df, {
        "similarity_definition": "word 1-2gram sublinear TF-IDF fit once on all non-empty normalised texts",
        "near_threshold": near_threshold,
        "grouping_margin": grouping_margin,
        "float_tolerance": NEAR_EPS,
        "n_pairs_at_or_above_threshold": len(all_pairs),
        "n_pairs_that_merged_groups": len(merging),
        "merging_pairs_sample": [[a, b, round(v, 6)] for a, b, v in sorted(merging, key=lambda x: -x[2])[:50]],
        "n_groups_before": n_before,
        "n_groups_after": int(df[GROUP_COL].nunique()),
        "largest_group_size": int(sizes.max()),
        "cross_split_pairs_after_split": 0,
    }

# --------------------------------------------------------------------------- persistence
def _ordered_columns(df: pd.DataFrame) -> list[str]:
    head = [ID_COL, GROUP_COL, LABEL_COL, "text"]
    rest = [c for c in [*TEXT_FIELDS, *METADATA_FIELDS] if c in df.columns]
    return head + rest


def save_splits(df: pd.DataFrame, out_dir: str | Path) -> dict[str, dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {}
    for name in SPLIT_NAMES:
        part = df[df[SPLIT_COL] == name].sort_values(ID_COL)
        path = out_dir / f"{name}.csv"
        part[_ordered_columns(part)].to_csv(path, index=False)
        info[name] = {
            "file": path.name,
            "n_rows": int(len(part)),
            "n_fraud": int((part[LABEL_COL] == 1).sum()),
            "fraud_rate": float((part[LABEL_COL] == 1).mean()) if len(part) else None,
            "sha256": file_sha256(path),
        }
    return info


def load_split(name: str, processed_dir: str | Path, final_evaluation: bool = False) -> pd.DataFrame:
    """Load a split and verify its hash against the manifest. 'test' requires final_evaluation=True."""
    if name not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}")
    if name == "test" and not final_evaluation:
        raise PermissionError(
            "The test split is locked. Pass final_evaluation=True only in the single final evaluation run "
            "(never for training, tuning, threshold selection, retrieval indexes or model selection)."
        )
    processed_dir = Path(processed_dir)
    manifest = json.loads((processed_dir / "split_manifest.json").read_text(encoding="utf-8"))
    path = processed_dir / manifest["splits"][name]["file"]
    if file_sha256(path) != manifest["splits"][name]["sha256"]:
        raise RuntimeError(f"{path} does not match the hash in split_manifest.json; re-obtain the shared split files.")
    return pd.read_csv(path)


# --------------------------------------------------------------------------- orchestration
def run_pipeline(
    raw_path: str | Path,
    processed_dir: str | Path,
    near_threshold: float = DEFAULT_NEAR_THRESHOLD,
    fractions: tuple[float, float, float] = DEFAULT_FRACTIONS,
    seed: int = SEED,
    inspect_only: bool = False,
    grouping_margin: float = 0.0,
) -> dict:
    raw_path = Path(raw_path)
    raw = load_raw(raw_path)
    schema = validate_schema(raw)
    df = add_model_text(raw)
    prof = profile(df)
    group, dup_stats = assign_duplicate_groups(df, near_threshold=near_threshold - grouping_margin)
    df[GROUP_COL] = group
    result: dict = {
        "raw_file": str(raw_path),
        "raw_sha256": file_sha256(raw_path),
        "schema": schema,
        "profile": prof,
        "duplicates": dup_stats,
    }
    if inspect_only:
        return result

    # NEW: merge groups of any cross-split pair the leakage check would flag, then re-split (threshold unchanged).
    df, resolution = resolve_cross_split_near_duplicates(
        df, fractions=fractions, seed=seed, near_threshold=near_threshold, grouping_margin=grouping_margin
    )
    result["near_duplicate_resolution"] = resolution
    parts = {name: df[df[SPLIT_COL] == name] for name in SPLIT_NAMES}
    leakage = check_leakage(parts, near_threshold=near_threshold)

    if not leakage["all_passed"]:
       raise RuntimeError(
           "Leakage checks failed; split files were not saved. "
           f"Details: {leakage}"
    )

    split_info = save_splits(df, processed_dir)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "fractions": list(fractions),
        "near_duplicate_threshold": near_threshold,
        "method": "group-aware stratified split; groups = exact/near-duplicate clusters; stratum = group has any fraud",
        "raw_file": raw_path.name,
        "raw_sha256": result["raw_sha256"],
        "splits": split_info,
        "near_duplicate_resolution": resolution,
        "pipeline_code_sha256": file_sha256(__file__),
        "leakage": {"all_passed": leakage["all_passed"], "checks": {k: v["passed"] for k, v in leakage["checks"].items()}},
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "scipy": scipy.__version__,
        },
    }
    (Path(processed_dir) / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    result.update({"split": split_info, "leakage": leakage, "manifest": manifest})
    return result
