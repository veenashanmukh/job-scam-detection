"""Demo-example route and the script's token selection, on temp files and a tiny synthetic graph."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from flask import Flask

pytest.importorskip("networkx")
from backend.routes import job_routes  # noqa: E402
from backend.services import entity_graph as eg  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import build_demo_examples as bd  # noqa: E402

REAL_TEXT = "Genuine Dataset Wording about a Real Employer"


@pytest.fixture()
def client():
    app = Flask(__name__)
    app.register_blueprint(job_routes.job_bp)
    return app.test_client()


def test_route_returns_empty_list_when_file_missing(client, monkeypatch, tmp_path):
    monkeypatch.setattr(job_routes, "DEMO_EXAMPLES_PATH", tmp_path / "none.json")
    assert client.get("/api/demo-examples").get_json() == []


def test_route_returns_the_five_fields_and_note_only(client, monkeypatch, tmp_path):
    path = tmp_path / "ex.json"
    path.write_text(json.dumps([{"title": "T", "company_profile": "", "description": "D", "requirements": "R",
                                 "benefits": "B", "note": "N", "job_id": 7, "secret": "x"}, "junk"]),
                    encoding="utf-8")
    monkeypatch.setattr(job_routes, "DEMO_EXAMPLES_PATH", path)
    assert client.get("/api/demo-examples").get_json() == [
        {"title": "T", "company_profile": "", "description": "D", "requirements": "R", "benefits": "B", "note": "N"}]


def test_route_ignores_a_corrupt_file(client, monkeypatch, tmp_path):
    path = tmp_path / "ex.json"
    path.write_text("not json", encoding="utf-8")
    monkeypatch.setattr(job_routes, "DEMO_EXAMPLES_PATH", path)
    assert client.get("/api/demo-examples").get_json() == []


def _posting(job_id, label, *tokens):
    return {"job_id": job_id, "fraudulent": label, "description": f"{REAL_TEXT} " + " ".join(tokens)}


def _graph():
    rows = []
    # Qualifying: only fraud links. Fewer links is less hub-like.
    rows += [_posting(i, 1, f"#EMAIL_{'a' * 16}#") for i in (1, 2)]                       # 2 fraud
    rows += [_posting(i, 1, f"#PHONE_{'b' * 16}#") for i in (3, 4, 5)]                    # 3 fraud
    rows += [_posting(i, 1, f"#URL_{'c' * 16}#") for i in (6, 7, 8, 9)]                   # 4 fraud
    rows += [_posting(i, 1, f"#EMAIL_{'d' * 16}#") for i in (10, 11, 12)] + [_posting(13, 0, f"#EMAIL_{'d' * 16}#")]
    rows += [_posting(14, 1, f"#EMAIL_{'e' * 16}#")]                                      # only 1 fraud
    rows += [_posting(15, 0, f"#EMAIL_{'f' * 16}#"), _posting(16, 0, f"#EMAIL_{'f' * 16}#")]  # legitimate only
    return eg.build_graph(pd.DataFrame(rows), excluded_job_ids=[])


def test_tokens_need_two_fraud_links_and_no_legitimate_link_and_are_ordered_least_hub_first():
    tokens = bd.pick_contact_tokens(_graph(), 5)
    assert tokens == [f"#EMAIL_{'a' * 16}#", f"#PHONE_{'b' * 16}#", f"#URL_{'c' * 16}#"]
    assert bd.pick_contact_tokens(_graph(), 5) == tokens  # deterministic
    assert bd.pick_contact_tokens(_graph(), 2) == tokens[:2]


def test_examples_use_one_distinct_token_each_and_copy_no_dataset_text():
    examples = bd.build_examples(_graph())
    fraud, legit = examples[:3], examples[3]
    used = [next(t for t in bd.pick_contact_tokens(_graph(), 5) if t in e["description"]) for e in fraud]
    assert len(set(used)) == 3
    assert all(f"Contact: {t}" in e["description"] for t, e in zip(used, fraud))
    assert all("{contact}" not in json.dumps(e) and e["note"] == bd.FRAUD_NOTE for e in fraud)
    assert legit["company_profile"].startswith("Northbridge Systems") and legit["note"] == bd.LEGIT_NOTE
    assert REAL_TEXT.lower() not in json.dumps(examples).lower()
    assert all(set(e) == {*bd.FRAUD_EXAMPLES[0], "note"} for e in examples)


def test_fewer_qualifying_tokens_means_fewer_fraud_examples():
    graph = eg.build_graph(pd.DataFrame([_posting(1, 1, f"#EMAIL_{'a' * 16}#"), _posting(2, 1, f"#EMAIL_{'a' * 16}#")]),
                           excluded_job_ids=[])
    examples = bd.build_examples(graph)
    assert len(examples) == 2 and examples[-1]["note"] == bd.LEGIT_NOTE
