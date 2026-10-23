"""NEW tests: one corpus-fit similarity shared by grouping and the leakage check; deterministic single-pass resolver."""
import json

import numpy as np
import pandas as pd
import pytest

from backend.services import dataset as ds
from backend.tests.conftest import make_fixture


def _df_with_planted_pairs(n_pairs=15):
    """Fixture + near-duplicate copies, each row its own group, so a split can separate the pairs
    (simulates grouping that missed pairs)."""
    df = ds.add_model_text(make_fixture())
    extra = []
    for k in range(n_pairs):
        r = df.iloc[50 + k].copy()
        r[ds.ID_COL] = 90000 + k
        r["text"] = r["text"] + " apply today"
        extra.append(r)
    df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    df[ds.GROUP_COL] = df[ds.ID_COL]
    return df


def _parts(df):
    return {n: df[df[ds.SPLIT_COL] == n] for n in ds.SPLIT_NAMES}


def test_threshold_is_not_weakened():
    assert ds.DEFAULT_NEAR_THRESHOLD == 0.90


def test_setup_really_leaks_before_resolution():
    df = _df_with_planted_pairs()
    df[ds.SPLIT_COL] = ds.group_stratified_split(df)
    assert not ds.check_leakage(_parts(df))["all_passed"]  # guards against a vacuous test


def test_resolution_is_single_pass_and_passes_unchanged_gate():
    out, res = ds.resolve_cross_split_near_duplicates(_df_with_planted_pairs())
    assert res["cross_split_pairs_after_split"] == 0
    assert res["n_pairs_that_merged_groups"] >= 15
    assert ds.check_leakage(_parts(out), near_threshold=0.90)["all_passed"]
    by_id = out.set_index(ds.ID_COL)[ds.SPLIT_COL]
    assert all(by_id[90000 + k] == by_id[1050 + k] for k in range(15))


def test_pair_similarity_does_not_depend_on_other_rows_in_the_splits():
    """Root cause of the old non-convergence: similarity moved with the split. It must not any more."""
    df = _df_with_planted_pairs(1)
    a, b = int(df[ds.ID_COL].iloc[-1]), 1050
    def sim_with(other_rows_to):
        d = df.copy()
        d[ds.SPLIT_COL] = "train"
        d.loc[d[ds.ID_COL] == a, ds.SPLIT_COL] = "val"
        movable = d[(d[ds.ID_COL] != a) & (d[ds.ID_COL] != b)].index[:150]
        d.loc[movable, ds.SPLIT_COL] = other_rows_to
        d.loc[d[ds.ID_COL] == b, ds.SPLIT_COL] = "train"
        pairs = ds.find_cross_split_near_pairs(_parts(d))
        return {(x, y): s for x, y, s in pairs}[(a, b)]
    assert sim_with("train") == pytest.approx(sim_with("test"), abs=1e-12)
    assert sim_with("train") == pytest.approx(sim_with("val"), abs=1e-12)


def test_grouping_and_check_use_the_same_definition(fixture_df):
    """Every pair the grouping would merge is exactly what the cross-split finder reports if separated."""
    df = ds.add_model_text(fixture_df)
    norm = df["text"].map(ds.normalize_for_dedupe).tolist()
    pos = [i for i, t in enumerate(norm) if t]
    group_pairs = {(int(df[ds.ID_COL].iloc[pos[a]]), int(df[ds.ID_COL].iloc[pos[b]]))
                   for a, b, _ in ds.near_duplicate_pairs([norm[i] for i in pos])}
    df[ds.SPLIT_COL] = np.where(np.arange(len(df)) % 2 == 0, "train", "val")  # arbitrary split
    parts = _parts(df)
    found = {tuple(sorted((a, b))) for a, b, _ in ds.find_cross_split_near_pairs(parts)}
    expected = {tuple(sorted(p)) for p in group_pairs
                if df.set_index(ds.ID_COL)[ds.SPLIT_COL][p[0]] != df.set_index(ds.ID_COL)[ds.SPLIT_COL][p[1]]}
    assert found == expected


def test_pair_exactly_on_threshold_is_flagged():
    texts = ["alpha beta gamma delta epsilon", "alpha beta gamma delta zeta"]
    X = ds._tfidf(texts)
    s = float((X[0] @ X[1].T).toarray()[0, 0])
    assert ds.near_duplicate_pairs(texts, threshold=s) != []          # boundary included (float tolerance)
    assert ds.near_duplicate_pairs(texts, threshold=s + 1e-6) == []   # but not above it


def test_resolver_is_deterministic_and_row_order_invariant():
    df = _df_with_planted_pairs()
    out1, _ = ds.resolve_cross_split_near_duplicates(df)
    out2, _ = ds.resolve_cross_split_near_duplicates(df.sample(frac=1.0, random_state=3).reset_index(drop=True))
    m1 = out1.set_index(ds.ID_COL)[[ds.GROUP_COL, ds.SPLIT_COL]].sort_index()
    m2 = out2.set_index(ds.ID_COL)[[ds.GROUP_COL, ds.SPLIT_COL]].sort_index()
    pd.testing.assert_frame_equal(m1, m2)


def test_clean_corpus_merges_nothing(fixture_df):
    df = ds.add_model_text(fixture_df)
    df[ds.GROUP_COL], _ = ds.assign_duplicate_groups(df)
    before = df[ds.GROUP_COL].copy()
    out, res = ds.resolve_cross_split_near_duplicates(df)
    assert res["n_pairs_that_merged_groups"] == 0
    assert out[ds.GROUP_COL].equals(before)


def test_merge_is_transitive_and_uses_smallest_id():
    df = pd.DataFrame({ds.ID_COL: [1, 2, 3, 4], ds.GROUP_COL: [1, 2, 3, 4]})
    out = ds.merge_groups_by_pairs(df, [(2, 1, 0.95), (3, 2, 0.95)])
    assert out.tolist() == [1, 1, 1, 4]


def test_oversized_merge_raises_instead_of_saving():
    with pytest.raises(ds.GroupTooLargeError):
        ds.resolve_cross_split_near_duplicates(_df_with_planted_pairs(), max_group_fraction=0.0001)


def test_gate_still_flags_a_planted_cross_split_pair(fixture_df):
    df = ds.add_model_text(fixture_df)
    df[ds.GROUP_COL], _ = ds.assign_duplicate_groups(df)
    df[ds.SPLIT_COL] = ds.group_stratified_split(df)
    parts = _parts(df)
    src = parts["train"].iloc[5].copy()
    src[ds.ID_COL] = src[ds.GROUP_COL] = 999999
    src["text"] += " one more word"
    parts["test"] = pd.concat([parts["test"], src.to_frame().T], ignore_index=True)
    rep = ds.check_leakage(parts)
    assert not rep["checks"]["no_cross_split_near_duplicates"]["passed"]


def test_pipeline_records_resolution_and_code_hash(fixture_csv, tmp_path):
    out = tmp_path / "processed"
    res = ds.run_pipeline(fixture_csv, out)
    assert res["leakage"]["all_passed"]
    m = json.loads((out / "split_manifest.json").read_text())
    assert m["near_duplicate_resolution"]["cross_split_pairs_after_split"] == 0
    assert m["near_duplicate_threshold"] == 0.90
    assert m["pipeline_code_sha256"] == ds.file_sha256(ds.__file__)


def test_grouping_margin_keeps_sub_threshold_pairs_together_but_gate_stays_at_090():
    df = ds.add_model_text(make_fixture())
    extra = []
    for k in range(15):
        r = df.iloc[50 + k].copy()
        r[ds.ID_COL] = 90000 + k
        r["text"] = r["text"] + " " + " ".join(f"extra{k}x{j}" for j in range(40))  # clearly < 0.90 similar
        extra.append(r)
    df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    df[ds.GROUP_COL] = df[ds.ID_COL]
    norm = df["text"].map(ds.normalize_for_dedupe).tolist()
    ids = df[ds.ID_COL].to_numpy()
    def planted(th):
        return {(int(ids[a]), int(ids[b])) for a, b, _ in ds.near_duplicate_pairs(norm, th)
                if max(ids[a], ids[b]) >= 90000}
    assert planted(0.90) == set()                                       # setup: planted pairs are NOT duplicates at 0.90
    sims = planted(0.5)
    assert len(sims) >= 15                                              # ...but are similar enough for margin 0.5
    out, res = ds.resolve_cross_split_near_duplicates(df, grouping_margin=0.5)
    by_id = out.set_index(ds.ID_COL)[ds.SPLIT_COL]
    assert sims and all(by_id[90000 + k] == by_id[1050 + k] for k in range(15))
    assert res["near_threshold"] == 0.90 and res["grouping_margin"] == 0.5
    assert ds.check_leakage(_parts(out), near_threshold=0.90)["all_passed"]


def test_grouping_margin_validation():
    with pytest.raises(ValueError):
        ds.resolve_cross_split_near_duplicates(_df_with_planted_pairs(1), grouping_margin=-0.1)


def _run_verify(args):
    import subprocess, sys
    from pathlib import Path
    root = Path(ds.__file__).resolve().parents[2]
    return subprocess.run([sys.executable, str(root / "scripts" / "verify_splits.py"), *args],
                          capture_output=True, text=True)


def test_verify_script_passes_fresh_output_and_detects_tampering_and_staleness(fixture_csv, tmp_path):
    out = tmp_path / "cand"
    ds.run_pipeline(fixture_csv, out)
    ok = _run_verify(["--dir", str(out), "--raw", str(fixture_csv), "--max-age-minutes", "60"])
    assert ok.returncode == 0, ok.stdout + ok.stderr

    # stale: manifest claims different code
    m = json.loads((out / "split_manifest.json").read_text())
    m["pipeline_code_sha256"] = "0" * 64
    (out / "split_manifest.json").write_text(json.dumps(m))
    assert _run_verify(["--dir", str(out), "--raw", str(fixture_csv)]).returncode == 5

    # tamper: a split file changes
    m["pipeline_code_sha256"] = ds.file_sha256(ds.__file__)
    (out / "split_manifest.json").write_text(json.dumps(m))
    with open(out / "train.csv", "a", encoding="utf-8") as f:
        f.write("\n")
    assert _run_verify(["--dir", str(out), "--raw", str(fixture_csv)]).returncode == 5
