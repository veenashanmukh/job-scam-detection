"""Evidence step and train-only posting index; synthetic data, no network, no LLM."""
import pandas as pd
import pytest

pytest.importorskip("networkx")
from backend.services import entity_graph as eg  # noqa: E402
from backend.services import evidence as ev  # noqa: E402
from backend.services import posting_index as pi  # noqa: E402

FEE_TEXT = "Data entry from home. Pay a registration fee today to secure your place. Earn weekly."
SECRET_COMPANY = "Totally Real Holdings Ltd"


def _train_frame():
    rows = []
    for i in range(6):
        rows.append({"job_id": 100 + i, "fraudulent": 1, "company_profile": SECRET_COMPANY,
                     "description": f"Data entry from home {i}. Pay a registration fee today. " + "x" * 300,
                     "text": f"Company profile: {SECRET_COMPANY}\nDescription: Data entry from home {i}. "
                             "Pay a registration fee today to secure your place."})
    for i in range(6):
        rows.append({"job_id": 200 + i, "fraudulent": 0, "company_profile": "Acme",
                     "description": f"Software engineer {i}. Review code and ship features.",
                     "text": f"Description: Software engineer {i}. Review code and ship features with the team."})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def index():
    return pi.build_index(_train_frame(), excluded_job_ids=[999])


# --- posting index ------------------------------------------------------------

def test_index_refuses_excluded_ids():
    with pytest.raises(ValueError, match="train only"):
        pi.build_index(_train_frame(), excluded_job_ids=[100])


def test_search_returns_top_k_train_postings_with_labels(index):
    hits = pi.search(index, FEE_TEXT, top_k=5)
    assert len(hits) == 5
    assert all(h["label"] == 1 for h in hits)  # the fee postings are the closest
    assert [h["similarity"] for h in hits] == sorted((h["similarity"] for h in hits), reverse=True)
    assert all(0 < h["similarity"] <= 1 for h in hits)


def test_snippets_are_capped_and_never_include_the_company_profile(index):
    for hit in pi.search(index, FEE_TEXT, top_k=5):
        assert len(hit["snippet"]) <= pi.SNIPPET_CHARS
        assert SECRET_COMPANY.lower() not in hit["snippet"].lower()
        assert hit["snippet"].startswith("Data entry from home")


def test_search_with_no_overlap_or_empty_text_returns_nothing(index):
    assert pi.search(index, "zzzz qqqq") == []
    assert pi.search(index, "   ") == []


def test_index_cache_verifies_train_hash(index, tmp_path):
    path = tmp_path / "idx.pkl"
    pi.save_index(index, "a" * 64, path)
    assert pi.load_index("a" * 64, path).matrix.shape == index.matrix.shape
    with pytest.raises(ValueError, match="different train.csv"):
        pi.load_index("b" * 64, path)


# --- evidence step ------------------------------------------------------------

@pytest.fixture()
def graph():
    rows = [{"job_id": 100 + i, "fraudulent": int(i < 3), "description": f"Mail #EMAIL_{'e' * 32}#"} for i in range(4)]
    return eg.build_graph(pd.DataFrame(rows), excluded_job_ids=[])


def _guidance(text):
    return [{"id": "ftc_job_scams", "title": "Job scams", "publisher": "FTC", "url": "https://example.org",
             "excerpt": "Never pay to get a job.", "warning_sign_tags": ["upfront_payment"], "similarity": 0.7}]


RULES = [{"label": "Upfront payment", "matched_text": "registration fee"},
         {"label": "Pressure tactic", "matched_text": "today to secure your position"}]


def test_evidence_items_get_stable_ids(index, graph):
    fields = {"description": f"{FEE_TEXT} Contact #EMAIL_{'e' * 32}#"}
    block = ev.build_evidence(fields["description"], fields, RULES,
                              posting_index=index, graph=graph, guidance_fn=_guidance)

    assert [s["id"] for s in block["similar_postings"]] == ["S1", "S2", "S3", "S4", "S5"]
    assert [g["id"] for g in block["graph"]["items"]] == ["G1"]
    assert block["graph"]["items"][0]["fraud_count"] == 3 and block["graph"]["items"][0]["kind"] == "EMAIL"
    assert [k["id"] for k in block["guidance"]] == ["K1"]
    assert [r["id"] for r in block["rule_signs"]] == ["R1", "R2"]
    assert block["similar_status"] == block["graph_status"] == block["guidance_status"] == "ok"
    assert ev.evidence_ids(block) == ["S1", "S2", "S3", "S4", "S5", "G1", "K1", "R1", "R2"]


def test_missing_sources_are_reported_not_raised():
    block = ev.build_evidence(FEE_TEXT, {"description": FEE_TEXT}, RULES)
    assert block["similar_postings"] == [] and block["similar_status"] == "unavailable"
    assert block["graph"] == {"items": []} and block["graph_status"] == "unavailable"
    assert block["guidance"] == [] and block["guidance_status"] == "unavailable"
    assert [r["id"] for r in block["rule_signs"]] == ["R1", "R2"]


def test_failing_sources_never_fail_the_request(index, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("index exploded")

    monkeypatch.setattr(pi, "search", boom)
    monkeypatch.setattr(eg, "query", boom)
    block = ev.build_evidence(FEE_TEXT, {"description": FEE_TEXT}, RULES,
                              posting_index=index, graph=object(), guidance_fn=boom)
    assert block["similar_status"] == "error" and block["graph_status"] == "error"
    assert block["guidance_status"] == "unavailable"


def test_retrieval_is_independent_of_the_llm(index, monkeypatch):
    from backend.services import llm_explainer

    def no_llm(*args, **kwargs):
        raise AssertionError("evidence step must not touch the LLM")

    monkeypatch.setattr(llm_explainer, "client_from_env", no_llm)
    monkeypatch.setattr(llm_explainer, "explain", no_llm)
    block = ev.build_evidence(FEE_TEXT, {"description": FEE_TEXT}, RULES, posting_index=index)
    assert len(block["similar_postings"]) == 5


def test_graph_evidence_contains_hashed_ids_only(graph):
    raw = "jane@example.com"
    fields = {"description": f"Write to {raw} or #EMAIL_{'e' * 32}#"}
    block = ev.build_evidence(fields["description"], fields, [], graph=graph)
    assert raw not in str(block)
    assert block["graph"]["items"][0]["entity"] == f"EMAIL:{'e' * 32}"


# --- similarity threshold -----------------------------------------------------

def _fake_hits(similarities):
    return [{"job_id": i, "label": i % 2, "similarity": s, "snippet": f"posting {i}"}
            for i, s in enumerate(similarities)]


def test_similar_postings_below_threshold_are_omitted(monkeypatch):
    monkeypatch.setattr(pi, "search", lambda index, text, top_k: _fake_hits([0.29, 0.2, 0.1]))
    block = ev.build_evidence(FEE_TEXT, {"description": FEE_TEXT}, [], posting_index=object())
    assert block["similar_postings"] == [] and block["similar_status"] == "ok"
    assert block["similar_min_similarity"] == ev.MIN_SIMILARITY == 0.30


def test_only_items_at_or_above_threshold_are_kept_and_renumbered(monkeypatch):
    monkeypatch.setattr(pi, "search", lambda index, text, top_k: _fake_hits([0.62, 0.30, 0.299, 0.1]))
    block = ev.build_evidence(FEE_TEXT, {"description": FEE_TEXT}, [], posting_index=object())
    assert [(s["id"], s["similarity"]) for s in block["similar_postings"]] == [("S1", 0.62), ("S2", 0.30)]


# --- guidance floor -----------------------------------------------------------

def test_guidance_below_floor_or_without_similarity_is_omitted():
    records = [
        {"id": "a", "title": "Close", "similarity": 0.47},
        {"id": "b", "title": "At floor", "similarity": 0.25},
        {"id": "c", "title": "Weak", "similarity": 0.249},
    ]
    block = ev.build_evidence(FEE_TEXT, {"description": FEE_TEXT}, [], guidance_fn=lambda text: records)
    assert [(k["id"], k["title"]) for k in block["guidance"]] == [("K1", "Close"), ("K2", "At floor")]
    assert block["guidance_status"] == "ok" and block["guidance_min_similarity"] == ev.MIN_GUIDANCE_SIMILARITY == 0.25

    none = ev.build_evidence(FEE_TEXT, {"description": FEE_TEXT}, [],
                             guidance_fn=lambda text: [{"id": "x", "similarity": 0.1}, {"id": "y"}])
    assert none["guidance"] == [] and none["guidance_status"] == "ok"
    assert not any(i.startswith("K") for i in ev.evidence_ids(none))


# --- live company lookup and the "no links" note -----------------------------------

def _company_graph(n_postings=4, n_fraud=3, company="Acme Holdings"):
    rows = [{"job_id": 500 + i, "fraudulent": int(i < n_fraud), "company_profile": f"{company}\nWe do things.",
             "description": f"Role {i}"} for i in range(n_postings)]
    return eg.build_graph(pd.DataFrame(rows), excluded_job_ids=[])


def test_company_profile_match_reports_counts_without_company_text():
    fields = {"company_profile": "<p>ACME Holdings.</p> Second line", "description": "Some role"}
    block = ev.build_evidence("Some role", fields, [], graph=_company_graph())
    item = block["graph"]["items"][0]
    assert item["id"] == "G1" and item["kind"] == "COMPANY"
    assert item["message"] == "This company profile appears in 4 known postings (3 labelled fraud)."
    assert block["graph"]["note"] == ""
    assert "acme" not in str(block).lower()


def test_hub_company_is_capped_and_shows_no_link():
    graph = _company_graph(n_postings=eg.HUB_MAX_POSTINGS + 1, n_fraud=10)
    block = ev.build_evidence("x", {"company_profile": "Acme Holdings"}, [], graph=graph)
    assert block["graph"]["items"] == [] and block["graph"]["note"] == "No linked companies or contacts found."


def test_no_match_says_so_when_graph_is_available():
    block = ev.build_evidence("x", {"company_profile": "Unknown Co", "description": "x"}, [], graph=_company_graph())
    assert block["graph_status"] == "ok" and block["graph"]["note"] == "No linked companies or contacts found."


def test_graph_unavailable_has_no_note():
    block = ev.build_evidence("x", {"company_profile": "Acme Holdings"}, [], graph=None)
    assert block["graph"] == {"items": []} and block["graph_status"] == "unavailable"
