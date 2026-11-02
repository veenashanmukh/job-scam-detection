"""Entity-link evidence: each matched entity reports its own true N and M; examples are balanced. Synthetic graph only."""
import pandas as pd
import pytest

pytest.importorskip("networkx")
from backend.services import entity_graph as eg  # noqa: E402
from backend.services import evidence as ev  # noqa: E402

URL_BIG = "a" * 16
URL_B = "b" * 16
EMAIL_A = "c" * 16


def _graph(rows):
    return eg.build_graph(pd.DataFrame(rows), excluded_job_ids=[])


def _row(job_id, label, *tokens, company=None):
    return {"job_id": job_id, "fraudulent": label, "company_profile": company,
            "description": " ".join(tokens) or "no contact here"}


def _fields(*tokens, company=None):
    return {"description": " ".join(tokens), "company_profile": company}


# --- true totals, uncapped ----------------------------------------------------------------------------------------

def test_an_entity_with_forty_links_reports_forty_and_its_true_fraud_count():
    rows = [_row(i, int(i < 30), f"#URL_{URL_BIG}#") for i in range(40)]            # 30 fraud, 10 legitimate
    graph = _graph(rows)
    links = eg.entity_links(graph, _fields(f"#URL_{URL_BIG}#"))
    assert [(l["kind"], l["linked_total"], l["fraud_total"]) for l in links] == [("URL", 40, 30)]
    assert len(links[0]["examples"]) == eg.MAX_LINKED_RETURNED == 20                  # the cap limits only the examples
    block, status = ev._graph(graph, _fields(f"#URL_{URL_BIG}#"))
    item = block["items"][0]
    assert status == "ok" and (item["linked_count"], item["fraud_count"]) == (40, 30)
    assert item["message"] == "This website appears in 40 known postings (30 labelled fraud)."
    assert block["linked_postings_total"] == 40 and block["fraud_neighbour_count"] == 30


def test_shared_entities_each_report_their_own_counts_and_a_shared_posting_counts_under_each():
    rows = ([_row(i, 1, f"#URL_{URL_B}#", f"#EMAIL_{EMAIL_A}#") for i in range(3)]          # shared by both entities
            + [_row(10 + i, 1, f"#EMAIL_{EMAIL_A}#") for i in range(2)]                     # only the email
            + [_row(20 + i, int(i < 2), f"#URL_{URL_B}#") for i in range(5)])               # only the URL
    graph = _graph(rows)
    posting = _fields(f"#URL_{URL_B}#", f"#EMAIL_{EMAIL_A}#")
    block, _ = ev._graph(graph, posting)
    by_kind = {i["kind"]: i for i in block["items"]}
    assert (by_kind["EMAIL"]["linked_count"], by_kind["EMAIL"]["fraud_count"]) == (5, 5)    # 3 shared + 2 own
    assert (by_kind["URL"]["linked_count"], by_kind["URL"]["fraud_count"]) == (8, 5)        # 3 shared + 5 own, 2 of them fraud
    assert [i["id"] for i in block["items"]] == ["G1", "G2"] and block["items"][0]["kind"] == "EMAIL"   # email first
    assert block["linked_postings_total"] == 10                                              # the union is still counted once
    assert by_kind["URL"]["message"] == "This website appears in 8 known postings (5 labelled fraud)."


def test_company_profile_gets_its_own_item_even_when_a_contact_reaches_the_same_postings():
    rows = [_row(i, 1, f"#URL_{URL_B}#", company="Acme Holdings\nAbout us") for i in range(4)]
    graph = _graph(rows)
    block, _ = ev._graph(graph, _fields(f"#URL_{URL_B}#", company="ACME holdings."))
    kinds = {i["kind"]: (i["linked_count"], i["fraud_count"]) for i in block["items"]}
    assert kinds == {"URL": (4, 4), "COMPANY": (4, 4)}
    assert {i["kind_name"] for i in block["items"]} == {"website", "company profile"}


def test_a_hub_entity_is_not_in_the_graph_and_gets_no_item_and_unmatched_entities_are_skipped():
    rows = [_row(i, 0, f"#URL_{URL_B}#") for i in range(eg.HUB_MAX_POSTINGS + 1)] + [_row(900, 1, f"#EMAIL_{EMAIL_A}#")]
    graph = _graph(rows)
    block, _ = ev._graph(graph, _fields(f"#URL_{URL_B}#", f"#EMAIL_{EMAIL_A}#", f"#PHONE_{'d' * 16}#"))
    assert [(i["kind"], i["linked_count"]) for i in block["items"]] == [("EMAIL", 1)]
    none, _ = ev._graph(graph, _fields(f"#PHONE_{'d' * 16}#"))
    assert none["items"] == [] and none["note"] == "No linked companies or contacts found."


def test_excluding_a_posting_removes_it_from_every_entity_total():
    rows = [_row(i, 1, f"#URL_{URL_B}#") for i in range(5)]
    graph = _graph(rows)
    links = eg.entity_links(graph, _fields(f"#URL_{URL_B}#"), exclude_job_id=2)
    assert (links[0]["linked_total"], links[0]["fraud_total"]) == (4, 4)
    assert 2 not in [e["job_id"] for e in links[0]["examples"]]


# --- balanced examples ---------------------------------------------------------------------------------------------

def _pairs(n_fraud, n_legit):
    return [(i, 1) for i in range(n_fraud)] + [(1000 + i, 0) for i in range(n_legit)]


def _split(examples):
    return sum(label for _, label in examples), sum(1 - label for _, label in examples)


@pytest.mark.parametrize("n_fraud, n_legit, expected", [
    (30, 40, (10, 10)),      # both exist: half and half
    (25, 0, (20, 0)),        # only fraud: fill the list from it
    (0, 25, (0, 20)),
    (30, 3, (17, 3)),        # legitimate side short: the fraud side tops up
    (2, 30, (2, 18)),        # fraud side short: the legitimate side tops up
    (4, 5, (4, 5)),          # fewer than the cap: everything is shown
])
def test_examples_are_balanced_without_a_fraud_first_bias(n_fraud, n_legit, expected):
    chosen = eg.balanced_examples(_pairs(n_fraud, n_legit))
    assert _split(chosen) == expected
    assert chosen == sorted(chosen) and len(set(chosen)) == len(chosen)
    assert chosen == eg.balanced_examples(list(reversed(_pairs(n_fraud, n_legit))))   # stable whatever the input order


def test_the_overall_example_list_in_query_is_balanced_too():
    rows = [_row(i, 1, f"#URL_{URL_B}#") for i in range(25)] + [_row(100 + i, 0, f"#URL_{URL_B}#") for i in range(25)]  # 50: not a hub
    result = eg.query(_graph(rows), _fields(f"#URL_{URL_B}#"))
    assert _split([(p["job_id"], p["label"]) for p in result["linked_postings"]]) == (10, 10)
    assert result["linked_postings_total"] == 50 and result["fraud_neighbour_count"] == 25


# --- nothing that scores reads the capped example list --------------------------------------------------------------

def test_graph_features_do_not_depend_on_the_example_cap(monkeypatch):
    rows = [_row(i, int(i % 2), f"#URL_{URL_BIG}#") for i in range(40)]
    graph = _graph(rows)
    fields = _fields(f"#URL_{URL_BIG}#")
    before = eg.graph_features(graph, fields)
    monkeypatch.setattr(eg, "MAX_LINKED_RETURNED", 3)
    assert eg.graph_features(graph, fields) == before and before["ents_n"] == 40 and before["ents_fraud"] == 20
    assert eg.query(graph, fields)["linked_postings_total"] == 40


def test_scoring_code_and_the_final_test_script_do_not_read_the_evidence_counts():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    for relative in ("scripts/final_test_eval.py", "backend/services/job_analysis.py", "backend/services/distilbert_service.py"):
        source = (root / relative).read_text(encoding="utf-8")
        assert "linked_count" not in source and "linked_postings" not in source and "entity_links" not in source, relative
    combiner = (root / "scripts" / "eval_graph_combiner_val.py").read_text(encoding="utf-8")
    assert "graph_features" in combiner and "linked_postings" not in combiner and "linked_count" not in combiner
