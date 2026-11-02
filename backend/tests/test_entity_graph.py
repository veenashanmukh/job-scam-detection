"""Entity graph tests on synthetic postings only."""
import json

import pandas as pd
import pytest

pytest.importorskip("networkx")
from backend.services import dataset as ds  # noqa: E402
from backend.services import entity_graph as eg  # noqa: E402

EMAIL = "a" * 32
PHONE = "b" * 32
URL = "c" * 20
SHARED_EMAIL = "d" * 32


def _row(job_id, label, **fields):
    return {"job_id": job_id, "fraudulent": label, "title": "Clerk", "company_profile": None,
            "description": None, "requirements": None, "benefits": None, **fields}


# --- extract_entities ---------------------------------------------------------

def test_extract_entities_reads_tokens_from_the_four_fields_only():
    entities = eg.extract_entities({
        "title": f"#EMAIL_{'e' * 32}#",  # title is not an entity field
        "company_profile": f"<p>Acme Ltd &amp; Co.</p><p>Visit #URL_{URL}#</p>",
        "description": f"Mail #EMAIL_{EMAIL}# or #EMAIL_{EMAIL}#",
        "requirements": f"Call #PHONE_{PHONE}#",
        "benefits": "#URL_short#",  # fewer than 16 hex characters: ignored
    })
    assert entities == {"url": [URL], "email": [EMAIL], "phone": [PHONE], "company": "acme ltd & co"}


@pytest.mark.parametrize("profile, expected", [
    (None, None), ("", None), ("   ", None), ("<p> </p>", None), (float("nan"), None),
    ("  <h3>ACME Recruiting!</h3>\nMore text", "acme recruiting"),
    (f"#URL_{URL}#\nBeta Corp", "beta corp"),
])
def test_company_name_is_first_line_normalised_and_only_when_present(profile, expected):
    assert eg.extract_entities({"company_profile": profile})["company"] == expected


# --- hub filtering ------------------------------------------------------------

def test_entities_linked_to_more_than_50_postings_are_dropped():
    hub, kept = "1" * 32, "2" * 32
    rows = [_row(i, 0, description=f"#EMAIL_{hub}#") for i in range(eg.HUB_MAX_POSTINGS + 1)]
    rows += [_row(1000 + i, 1, description=f"#EMAIL_{kept}#") for i in range(eg.HUB_MAX_POSTINGS)]
    graph = eg.build_graph(pd.DataFrame(rows), excluded_job_ids=[999_999])

    assert f"EMAIL:{hub}" not in graph and f"EMAIL:{kept}" in graph
    assert graph.degree(f"EMAIL:{kept}") == eg.HUB_MAX_POSTINGS
    assert graph.graph["hubs_removed"] == 1
    assert eg.query(graph, {"description": f"#EMAIL_{hub}#"})["linked_postings_total"] == 0


# --- train-only guard ---------------------------------------------------------

def test_build_graph_refuses_frames_containing_excluded_job_ids():
    frame = pd.DataFrame([_row(1, 0), _row(2, 1)])
    with pytest.raises(ValueError, match="train only"):
        eg.build_graph(frame, excluded_job_ids=[2, 3])
    assert eg.build_graph(frame, excluded_job_ids=[3]).number_of_nodes() == 2


def test_build_graph_requires_the_exclusion_list():
    with pytest.raises(TypeError):
        eg.build_graph(pd.DataFrame([_row(1, 0)]))


# --- cache hash verification --------------------------------------------------

def test_cache_load_verifies_train_hash(tmp_path):
    graph = eg.build_graph(pd.DataFrame([_row(1, 1, description=f"#EMAIL_{EMAIL}#")]), excluded_job_ids=[])
    path = tmp_path / "graph.pkl"
    eg.save_graph(graph, "f" * 64, path)

    assert eg.load_graph("f" * 64, path).has_node(f"EMAIL:{EMAIL}")
    with pytest.raises(ValueError, match="different train.csv"):
        eg.load_graph("0" * 64, path)


def test_build_or_load_rejects_train_file_not_matching_manifest_and_rebuilds_stale_cache(tmp_path):
    train = pd.DataFrame([_row(1, 1, description=f"#EMAIL_{EMAIL}#"), _row(2, 0)])
    val = pd.DataFrame([_row(3, 0)])
    train.to_csv(tmp_path / "train.csv", index=False)
    val.to_csv(tmp_path / "val.csv", index=False)
    manifest = {"splits": {"train": {"file": "train.csv", "sha256": ds.file_sha256(tmp_path / "train.csv")},
                           "val": {"file": "val.csv", "sha256": ds.file_sha256(tmp_path / "val.csv")},
                           "test": {"file": "test.csv", "sha256": "0" * 64}}}  # no test.csv exists
    (tmp_path / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    cache = tmp_path / "graph.pkl"

    eg.save_graph(eg.build_graph(train.iloc[:1], excluded_job_ids=[]), "stale", cache)
    graph = eg.build_or_load(tmp_path, cache)
    assert graph.graph["n_train"] == 2  # stale cache was rebuilt
    assert eg.load_graph(manifest["splits"]["train"]["sha256"], cache).graph["n_train"] == 2

    manifest["splits"]["train"]["sha256"] = "0" * 64
    (tmp_path / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        eg.build_or_load(tmp_path, cache)


# --- query output -------------------------------------------------------------

@pytest.fixture()
def small_graph():
    rows = [
        _row(1, 1, description=f"#EMAIL_{SHARED_EMAIL}#", company_profile="Shady Jobs Inc\nWe pay well"),
        _row(2, 1, requirements=f"#PHONE_{PHONE}#", company_profile="Shady Jobs Inc"),
        _row(3, 0, description=f"#EMAIL_{SHARED_EMAIL}#"),
        _row(4, 0, benefits=f"#URL_{URL}#"),
    ]
    return eg.build_graph(pd.DataFrame(rows), excluded_job_ids=[99])


def test_query_counts_neighbours_and_smooths_the_rate(small_graph):
    posting = {"description": f"Email #EMAIL_{SHARED_EMAIL}#", "company_profile": "SHADY JOBS INC."}
    result = eg.query(small_graph, posting)

    assert result["linked_postings_total"] == 3
    assert result["fraud_neighbour_count"] == 2
    assert result["fraud_rate_smoothed"] == pytest.approx((2 + 0.0475 * eg.PRIOR_WEIGHT) / (3 + eg.PRIOR_WEIGHT))
    assert [p["job_id"] for p in result["linked_postings"]] == [1, 2, 3]  # fraud first
    assert result["linked_postings"][0]["shared_entity"] == f"EMAIL:{SHARED_EMAIL}"
    assert {p["label"] for p in result["linked_postings"]} == {0, 1}


def test_query_with_no_links_returns_the_prior(small_graph):
    result = eg.query(small_graph, {"description": "nothing to see"})
    assert result["entities_found"] == [] and result["linked_postings"] == []
    assert result["fraud_rate_smoothed"] == pytest.approx(eg.FRAUD_PRIOR)


def test_query_can_exclude_the_posting_itself(small_graph):
    result = eg.query(small_graph, {"description": f"#EMAIL_{SHARED_EMAIL}#"}, exclude_job_id=1)
    assert [p["job_id"] for p in result["linked_postings"]] == [3]


def test_query_output_contains_no_raw_contact_details_or_profile_text(small_graph):
    raw_email, raw_phone = "jane.doe@example.com", "+1 555 010 9999"
    posting = {
        "company_profile": "Shady Jobs Inc\nContact jane.doe@example.com",
        "description": f"Write to {raw_email} or call {raw_phone}. #EMAIL_{SHARED_EMAIL}#",
        "requirements": f"#PHONE_{PHONE}#",
    }
    output = json.dumps(eg.query(small_graph, posting)).lower()

    for secret in (raw_email, raw_phone, "jane", "shady jobs", "example.com"):
        assert secret.lower() not in output
    for entity in eg.query(small_graph, posting)["entities_found"]:
        kind, digest = entity.split(":", 1)
        assert kind in eg.ENTITY_KINDS and all(c in "0123456789abcdef" for c in digest)


def test_graph_features_split_contact_entities(small_graph):
    posting = {"description": f"#EMAIL_{SHARED_EMAIL}#", "company_profile": "Shady Jobs Inc"}
    feats = eg.graph_features(small_graph, posting)
    assert feats["ents_n"] == 3 and feats["ents_fraud"] == 2
    assert feats["emailphone_n"] == 2 and feats["emailphone_fraud"] == 1
