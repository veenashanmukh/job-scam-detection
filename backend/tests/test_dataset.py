import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from backend.services import dataset as ds
from backend.services import dq_report
from backend.tests.conftest import ROOT, make_fixture


# ------------------------------------------------------------------ schema
def test_schema_accepts_fixture(fixture_df):
    rep = ds.validate_schema(fixture_df)
    assert rep["n_rows"] == len(fixture_df)


@pytest.mark.parametrize("col", ["job_id", "fraudulent", "description", "title"])
def test_schema_rejects_missing_required_column(fixture_df, col):
    with pytest.raises(ds.SchemaError):
        ds.validate_schema(fixture_df.drop(columns=[col]))


def test_schema_rejects_duplicate_ids(fixture_df):
    bad = fixture_df.copy()
    bad.loc[1, "job_id"] = bad.loc[0, "job_id"]
    with pytest.raises(ds.SchemaError, match="unique"):
        ds.validate_schema(bad)


def test_schema_rejects_non_binary_or_missing_label(fixture_df):
    bad = fixture_df.copy()
    bad.loc[0, "fraudulent"] = 2
    with pytest.raises(ds.SchemaError, match="binary"):
        ds.validate_schema(bad)
    bad = fixture_df.copy()
    bad.loc[0, "fraudulent"] = np.nan
    with pytest.raises(ds.SchemaError, match="missing"):
        ds.validate_schema(bad)


def test_schema_warns_on_absent_metadata(fixture_df):
    rep = ds.validate_schema(fixture_df.drop(columns=["industry"]))
    assert any("industry" in w for w in rep["warnings"])


def test_load_raw_missing_file_has_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="README"):
        ds.load_raw(tmp_path / "nope.csv")


# ------------------------------------------------------------------ text construction
def test_build_text_does_not_invent_missing_fields():
    row = pd.Series({"title": "Cook", "company_profile": None, "description": "<p>Make&nbsp;food</p>",
                     "requirements": "  ", "benefits": np.nan})
    text = ds.build_text(row)
    assert text == "Title: Cook\nDescription: Make food"
    assert "Company profile" not in text and "Benefits" not in text and "Requirements" not in text


def test_build_text_all_empty_gives_empty_string():
    row = pd.Series({f: None for f in ds.TEXT_FIELDS})
    assert ds.build_text(row) == ""


def test_add_model_text_counts_present_fields(fixture_df):
    out = ds.add_model_text(fixture_df.head(3))
    assert {"text", "n_text_fields_present"} <= set(out.columns)
    assert (out["n_text_fields_present"] <= 5).all()


# ------------------------------------------------------------------ profile
def test_profile_label_counts_and_missingness(fixture_df):
    df = ds.add_model_text(fixture_df)
    p = ds.profile(df)
    assert sum(p["label_counts"].values()) == len(df)
    assert p["missingness"]["benefits"]["missing"] > 0
    assert p["empty_text_rows"] == 2


def test_profile_detects_redaction_tokens_and_contacts():
    df = pd.DataFrame({
        "job_id": [1, 2], "fraudulent": [0, 1],
        "title": ["A", "B"], "company_profile": [None, None],
        "description": ["Apply #EMAIL_abc123def# now", "mail bob@example.com or visit https://x.example/apply"],
        "requirements": [None, None], "benefits": [None, None],
    })
    p = ds.profile(ds.add_model_text(df))["contact_patterns"]
    assert p["rows_with_placeholder_tokens"] == 1
    assert p["rows_with_email_like"] == 1
    assert p["rows_with_url_like"] == 1


# ------------------------------------------------------------------ duplicates
@pytest.fixture(scope="module")
def grouped():
    df = ds.add_model_text(make_fixture())
    group, stats = ds.assign_duplicate_groups(df)
    df[ds.GROUP_COL] = group
    return df, stats


def test_exact_duplicates_share_group(grouped):
    df, _ = grouped
    for src_id, dup_ids in ((1005, (5000, 5001)), (1011, (5002, 5003)), (1017, (5004, 5005))):
        gids = df.set_index("job_id").loc[[src_id, *dup_ids], ds.GROUP_COL]
        assert gids.nunique() == 1


def test_near_duplicate_shares_group(grouped):
    df, _ = grouped
    gids = df.set_index("job_id").loc[[1023, 5006], ds.GROUP_COL]
    assert gids.nunique() == 1


def test_distinct_postings_are_not_grouped(grouped):
    df, stats = grouped
    # fixture plants 3 exact-dup groups, 1 near-dup group, 1 conflict group = 5 multi-member groups
    assert stats["n_multi_member_groups"] == 5
    assert stats["largest_group_size"] == 3


def test_empty_text_rows_are_not_grouped_together(grouped):
    df, _ = grouped
    empties = df[df["text"] == ""]
    assert len(empties) == 2
    assert empties[ds.GROUP_COL].nunique() == 2


def test_label_conflict_group_is_reported(grouped):
    _, stats = grouped
    assert stats["groups_with_conflicting_labels"] == 1


def test_group_id_is_smallest_job_id(grouped):
    df, _ = grouped
    assert (df[ds.GROUP_COL] <= df[ds.ID_COL]).all()


# ------------------------------------------------------------------ split
@pytest.fixture(scope="module")
def split_df(grouped):
    df, _ = grouped
    out = df.copy()
    out[ds.SPLIT_COL] = ds.group_stratified_split(out)
    return out


def test_split_covers_all_rows_once(split_df):
    assert split_df[ds.SPLIT_COL].isin(ds.SPLIT_NAMES).all()


def test_no_group_crosses_splits(split_df):
    assert (split_df.groupby(ds.GROUP_COL)[ds.SPLIT_COL].nunique() == 1).all()


def test_split_proportions_close_to_target(split_df):
    frac = split_df[ds.SPLIT_COL].value_counts(normalize=True)
    for name, target in zip(ds.SPLIT_NAMES, ds.DEFAULT_FRACTIONS):
        assert abs(frac[name] - target) < 0.04, (name, frac[name])


def test_split_is_stratified_on_fraud(split_df):
    overall = split_df[ds.LABEL_COL].mean()
    for name in ds.SPLIT_NAMES:
        part = split_df[split_df[ds.SPLIT_COL] == name]
        assert (part[ds.LABEL_COL] == 1).sum() >= 1
        assert abs(part[ds.LABEL_COL].mean() - overall) < 0.04


def test_split_is_deterministic_and_seed_sensitive(grouped):
    df, _ = grouped
    a = ds.group_stratified_split(df, seed=42)
    b = ds.group_stratified_split(df, seed=42)
    c = ds.group_stratified_split(df, seed=43)
    assert a.equals(b)
    assert not a.equals(c)


def test_split_rejects_bad_fractions(grouped):
    df, _ = grouped
    with pytest.raises(ValueError):
        ds.group_stratified_split(df, fractions=(0.5, 0.2, 0.2))


# ------------------------------------------------------------------ leakage checks
def _parts(split_df):
    return {n: split_df[split_df[ds.SPLIT_COL] == n] for n in ds.SPLIT_NAMES}


def test_leakage_checks_pass_on_clean_split(split_df):
    rep = ds.check_leakage(_parts(split_df))
    assert rep["all_passed"], rep


def test_leakage_check_detects_planted_id_and_text_leak(split_df):
    parts = _parts(split_df)
    leaked = parts["train"].iloc[[0]].copy()
    parts["test"] = pd.concat([parts["test"], leaked])
    rep = ds.check_leakage(parts)
    assert not rep["all_passed"]
    assert not rep["checks"]["no_job_id_overlap"]["passed"]
    assert not rep["checks"]["no_exact_text_overlap"]["passed"]
    assert not rep["checks"]["no_cross_split_near_duplicates"]["passed"]


def test_leakage_check_detects_planted_near_duplicate(split_df):
    parts = _parts(split_df)
    src = parts["train"].iloc[5].copy()
    src[ds.ID_COL] = 999999
    src[ds.GROUP_COL] = 999999
    src["text"] = src["text"] + " one more word"
    parts["val"] = pd.concat([parts["val"], src.to_frame().T], ignore_index=True)
    rep = ds.check_leakage(parts)
    assert rep["checks"]["no_exact_text_overlap"]["passed"]
    assert not rep["checks"]["no_cross_split_near_duplicates"]["passed"]


# ------------------------------------------------------------------ persistence + test lock
def test_pipeline_end_to_end_writes_manifest_and_locks_test(fixture_csv, tmp_path):
    out = tmp_path / "processed"
    res = ds.run_pipeline(fixture_csv, out)
    assert res["leakage"]["all_passed"]
    manifest = json.loads((out / "split_manifest.json").read_text())
    assert manifest["seed"] == 42 and sum(i["n_rows"] for i in manifest["splits"].values()) == res["profile"]["n_rows"]

    train = ds.load_split("train", out)
    assert len(train) == manifest["splits"]["train"]["n_rows"]
    with pytest.raises(PermissionError):
        ds.load_split("test", out)
    test = ds.load_split("test", out, final_evaluation=True)
    assert len(test) == manifest["splits"]["test"]["n_rows"]
    assert set(train[ds.ID_COL]).isdisjoint(test[ds.ID_COL])


def test_load_split_detects_tampering(fixture_csv, tmp_path):
    out = tmp_path / "processed"
    ds.run_pipeline(fixture_csv, out)
    with open(out / "train.csv", "a", encoding="utf-8") as f:
        f.write("tampered\n")
    with pytest.raises(RuntimeError, match="hash"):
        ds.load_split("train", out)


def test_pipeline_files_are_byte_identical_across_runs(fixture_csv, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    ra, rb = ds.run_pipeline(fixture_csv, a), ds.run_pipeline(fixture_csv, b)
    for name in ds.SPLIT_NAMES:
        assert ra["split"][name]["sha256"] == rb["split"][name]["sha256"]


def test_inspect_only_writes_no_splits(fixture_csv, tmp_path):
    out = tmp_path / "processed"
    res = ds.run_pipeline(fixture_csv, out, inspect_only=True)
    assert "split" not in res and not out.exists()


# ------------------------------------------------------------------ report + CLI
def test_report_renders_and_contains_gates(fixture_csv, tmp_path):
    res = ds.run_pipeline(fixture_csv, tmp_path / "p")
    md = dq_report.render_markdown(res)
    assert "## Gates" in md and "Leakage: no_group_overlap" in md and "Licence verified" in md and "MANUAL" in md


def test_cli_runs_and_reports_missing_file_cleanly(tmp_path):
    cmd = [sys.executable, str(ROOT / "scripts/prepare_data.py"), "--raw", str(tmp_path / "missing.csv"),
           "--out", str(tmp_path / "o"), "--report-dir", str(tmp_path / "r")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 2 and "not found" in r.stderr


def test_cli_full_run_on_fixture(fixture_csv, tmp_path):
    cmd = [sys.executable, str(ROOT / "scripts/prepare_data.py"), "--raw", str(fixture_csv),
           "--out", str(tmp_path / "o"), "--report-dir", str(tmp_path / "r")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "r" / "data_quality_report.md").exists()
    assert (tmp_path / "o" / "split_manifest.json").exists()


def test_cli_schema_error_exit_code(tmp_path, fixture_df):
    p = tmp_path / "bad.csv"
    fixture_df.drop(columns=["description"]).to_csv(p, index=False)
    cmd = [sys.executable, str(ROOT / "scripts/prepare_data.py"), "--raw", str(p),
           "--out", str(tmp_path / "o"), "--report-dir", str(tmp_path / "r")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 3 and "description" in r.stderr
