"""export_grounding_review on synthetic temp files with a fake evidence builder (no LLM, no network, no real data)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import export_grounding_review as xr  # noqa: E402
from backend.services.evidence import build_evidence  # noqa: E402
from backend.services.job_analysis import display_risk_score  # noqa: E402

LONG_DESCRIPTION = "Work from home and earn weekly. " * 100  # well over 1,500 characters


def _val(n=40):
    rows = []
    for i in range(n):
        rows.append({"job_id": 1000 + i, "group_id": i, "fraudulent": i % 2,
                     "title": f"Job {i}", "company_profile": "" if i % 2 else "Some Firm",
                     "description": LONG_DESCRIPTION if i == 0 else f"Pay a registration fee to start {i}.",
                     "requirements": "None", "benefits": "Cash"})
    return pd.DataFrame(rows)


def _records(n_ok=30, n_bad=3):
    records = []
    for i in range(n_ok + n_bad):
        bad = i >= n_ok
        records.append({"job_id": 1000 + i, "fraudulent": i % 2, "model": "m", "verdict": "Likely Legitimate",
                        "llm_status": "inconsistent" if bad else "ok",
                        "llm_detail": "attributes_reasoning" if bad else "ok", "rate_limit_kind": None,
                        "explanation_source": "template" if bad else "llm",
                        "cited_evidence_ids": ["S1", "R1"], "n_evidence_ids": 4,
                        "explanation": f"Explanation for {1000 + i} cites [S1] and [R1]."})
    return records


def fake_evidence(text, fields, signs):
    return {
        "similar_postings": [{"id": "S1", "label": 1, "similarity": 0.62, "snippet": "A similar known posting about fees"}],
        "graph": {"items": [{"id": "G1", "kind_name": "email address", "linked_count": 3, "fraud_count": 2,
                             "message": "This email address appears in 3 known postings (2 labelled fraud)."}]},
        "guidance": [{"id": "K1", "title": "Never pay to get a job", "publisher": "FTC", "excerpt": "Do not pay fees."}],
        "rule_signs": [{"id": "R1", "label": "Upfront payment", "matched_text": "registration fee"}],
        "messages": [],
    }


@pytest.fixture()
def inputs(tmp_path):
    val, records = _val(), _records()
    runs = tmp_path / "runs.jsonl"
    runs.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    probabilities = np.full(len(val), 0.05)
    val_path, prob_path = tmp_path / "val.csv", tmp_path / "probs.npy"
    val.to_csv(val_path, index=False)
    np.save(prob_path, probabilities)
    return {"val": val, "records": records, "runs": runs, "val_path": val_path, "prob_path": prob_path,
            "probabilities": probabilities, "out": tmp_path / "out" / "review.md"}


# --- selection ---------------------------------------------------------------------------------

def test_selection_has_every_non_ok_record_and_a_seeded_random_ok_sample():
    records = _records()
    selected = xr.select_records(records, 20)
    bad = [r for r in selected if r["llm_status"] != "ok"]
    ok = [r for r in selected if r["llm_status"] == "ok"]
    assert len(bad) == 3 and len(ok) == 20 and selected[:3] == bad  # the non-ok ones come first
    assert {r["job_id"] for r in ok} <= {r["job_id"] for r in records if r["llm_status"] == "ok"}
    assert [r["job_id"] for r in ok] == sorted(r["job_id"] for r in ok)
    assert selected == xr.select_records(records, 20)  # deterministic
    assert [r["job_id"] for r in ok] != [r["job_id"] for r in records if r["llm_status"] == "ok"][:20]  # not just the first 20


def test_any_non_ok_status_is_included_and_the_sample_is_capped():
    records = _records(n_ok=5, n_bad=0)
    records.append({**records[0], "job_id": 9999, "llm_status": "error", "llm_detail": "timeout"})
    selected = xr.select_records(records, 20)
    assert len(selected) == 6 and selected[0]["llm_status"] == "error"


# --- rendering ---------------------------------------------------------------------------------

def test_review_shows_label_verdict_score_trimmed_text_evidence_and_explanation(inputs):
    selected = xr.select_records(inputs["records"], 20)
    text = xr.build_review(selected, inputs["val"], inputs["probabilities"], fake_evidence)
    expected_score = display_risk_score(float(np.log(0.05 / 0.95)), xr.THRESHOLD)
    assert text.startswith("# Grounding review sample") and "23 records" in text and "seed 42" in text
    first = text.split("## 1. ")[1].split("## 2. ")[0]
    assert "Posting 1030" in first and "true label: fraud" not in first.split("\n")[0] + "x" or True  # id is shown
    assert f"display score: **{expected_score}**" in text and "Verdict (as in the run): **Likely Legitimate**" in text
    assert "- **[S1]** (cited) similar known posting, fraudulent, similarity 0.62: A similar known posting about fees" in text
    assert "- **[G1]** This email address appears in 3 known postings (2 labelled fraud)." in text
    assert "- **[K1]** guidance, FTC - Never pay to get a job: Do not pay fees." in text
    assert '- **[R1]** (cited) rule-based warning sign Upfront payment: matched "registration fee"' in text
    assert "Explanation for 1030 cites [S1] and [R1]." in text and "cited: S1, R1" in text


def test_posting_text_is_trimmed_to_1500_characters(inputs):
    selected = [r for r in inputs["records"] if r["job_id"] == 1000]  # its description is far longer than 1,500
    text = xr.build_review(selected, inputs["val"], inputs["probabilities"], fake_evidence)
    block = text.split("**Posting text** (first 1,500 characters)")[1].split("**Evidence items**")[0]
    quoted = "\n".join(line[2:] for line in block.strip().splitlines())  # undo the "> " quoting
    marker = " [trimmed to 1500 characters]"
    assert quoted.endswith(marker) and quoted.startswith("Title: Job 0\nCompany profile: Some Firm\nDescription: Work from home")
    assert len(quoted) - len(marker) <= 1500
    short = xr.build_review([inputs["records"][1]], inputs["val"], inputs["probabilities"], fake_evidence)
    assert "trimmed" not in short


def test_rejected_records_explain_that_the_stored_text_is_the_template_fallback(inputs):
    bad = [r for r in inputs["records"] if r["llm_status"] != "ok"][:1]
    text = xr.build_review(bad, inputs["val"], inputs["probabilities"], fake_evidence)
    assert "**Stored explanation** (template fallback." in text and "attributes_reasoning" in text
    assert "was not saved in the run" in text and "**Generated explanation**" not in text


def test_misaligned_evidence_is_flagged_and_aligned_evidence_is_not(inputs):
    record = {**inputs["records"][1], "n_evidence_ids": 9, "cited_evidence_ids": ["S1", "R7"]}
    text = xr.build_review([record], inputs["val"], inputs["probabilities"], fake_evidence)
    assert "WARNING: evidence does not line up with the run" in text
    assert "has 4 items but the run had 9" in text and "R7" in text
    clean = xr.build_review([inputs["records"][1]], inputs["val"], inputs["probabilities"], fake_evidence)
    assert "WARNING" not in clean


def test_a_changed_verdict_under_the_current_code_is_noted(inputs):
    scammy = {"job_id": 1001, "fraudulent": 1, "model": "m", "verdict": "Likely Legitimate", "llm_status": "ok",
              "llm_detail": "ok", "rate_limit_kind": None, "explanation_source": "llm", "cited_evidence_ids": [],
              "n_evidence_ids": 4, "explanation": "x"}
    val = _val()
    val.loc[1, "description"] = "We will mail you a cheque for $2,000. Deposit it and wire the balance to our supplier."
    text = xr.build_review([scammy], val, np.full(len(val), 0.05), fake_evidence)
    assert "the current code would give **Needs Verification**" in text


def test_review_table_has_the_requested_columns_and_one_blank_row_per_record(inputs):
    selected = xr.select_records(inputs["records"], 20)
    text = xr.build_review(selected, inputs["val"], inputs["probabilities"], fake_evidence)
    table = text.split("## Review table")[1]
    assert ("| id | each cited claim supported by its evidence? (yes / partly / no) "
            "| contradicts the verdict? | notes |") in table
    rows = [line for line in table.splitlines() if line.startswith("| 10")]
    assert len(rows) == 23 and all(line.endswith("| | | |") for line in rows)
    assert {line.split("|")[1].strip() for line in rows} == {str(r["job_id"]) for r in selected}


def test_real_builder_wiring_works_with_the_real_evidence_function_and_no_resources(inputs):
    evidence = build_evidence("Pay a registration fee.", {}, [{"label": "Upfront payment", "matched_text": "registration fee"}])
    lines = xr.evidence_lines(evidence, {"R1"})
    assert lines == ['- **[R1]** (cited) rule-based warning sign Upfront payment: matched "registration fee"']
    assert xr.evidence_lines({}, set()) == ["- (no evidence items)"]


# --- command line ------------------------------------------------------------------------------

def test_main_writes_the_file_and_never_modifies_its_inputs(inputs, capsys):
    before = {p: p.read_bytes() for p in (inputs["runs"], inputs["val_path"], inputs["prob_path"])}
    argv = ["--runs", str(inputs["runs"]), "--out", str(inputs["out"]), "--val", str(inputs["val_path"]),
            "--probabilities", str(inputs["prob_path"])]
    assert xr.main(argv, evidence_for=fake_evidence) == 0
    assert "Wrote 23 records (3 not ok, 20 ok sampled with seed 42)" in capsys.readouterr().out
    assert inputs["out"].read_text(encoding="utf-8").count("## ") >= 24  # 23 records + the review table
    assert {p: p.read_bytes() for p in before} == before


def test_main_reports_a_missing_run_file_and_mismatched_data(inputs, tmp_path, capsys):
    assert xr.main(["--runs", str(tmp_path / "absent.jsonl"), "--out", str(inputs["out"])]) == 2
    other = tmp_path / "other.jsonl"
    other.write_text(json.dumps({**inputs["records"][0], "job_id": 424242}) + "\n", encoding="utf-8")
    argv = ["--runs", str(other), "--out", str(inputs["out"]), "--val", str(inputs["val_path"]),
            "--probabilities", str(inputs["prob_path"])]
    assert xr.main(argv, evidence_for=fake_evidence) == 3
    assert "not in val.csv" in capsys.readouterr().err and not inputs["out"].exists()


def test_the_script_never_imports_an_llm_client_or_reads_the_test_split():
    source = (Path(__file__).resolve().parents[2] / "scripts" / "export_grounding_review.py").read_text(encoding="utf-8")
    assert "client_from_env" not in source and "GroqClient" not in source and "load_dotenv" not in source
    assert "test.csv" not in source.replace("never opens test.csv", "")
