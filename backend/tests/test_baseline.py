"""Tests for backend/services/baseline.py (synthetic data only; says nothing about the real dataset)."""
import json

import numpy as np
import pandas as pd
import pytest

from backend.services import baseline as bl
from backend.services import dataset as ds
from backend.tests.conftest import make_fixture

SCAM = " wire transfer processing fee western union no interview needed"


def _signal_csv(tmp_path):
    raw = make_fixture()
    raw.loc[raw["fraudulent"] == 1, "description"] += SCAM
    p = tmp_path / "raw.csv"
    raw.to_csv(p, index=False)
    return p


def test_xy_handles_nan_text():
    X, y = bl.xy(pd.DataFrame({"text": ["a b", np.nan], ds.LABEL_COL: [0, 1]}))
    assert X == ["a b", ""] and y.tolist() == [0, 1]


def test_select_threshold_max_f1_and_tiebreak():
    y = np.array([0, 0, 0, 1, 1, 1]); s = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9])
    assert bl.select_threshold(y, s, "max_f1") == pytest.approx(0.6)  # perfectly separable -> highest perfect threshold


def test_select_threshold_recall_at_precision():
    y = np.array([0, 0, 1, 0, 1, 1]); s = np.array([0.1, 0.2, 0.3, 0.5, 0.8, 0.9])
    t = bl.select_threshold(y, s, "recall_at_precision", min_precision=1.0)
    m = bl.metrics_at_threshold(y, s, t)
    assert m["precision"] == 1.0 and m["recall"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        bl.select_threshold(np.array([0, 1, 0]), np.array([0.9, 0.1, 0.8]), "recall_at_precision", min_precision=0.99)
    with pytest.raises(ValueError):
        bl.select_threshold(y, s, "recall_at_precision")           # floor required
    with pytest.raises(ValueError):
        bl.select_threshold(y, s, "accuracy")                      # unknown rule


def test_metrics_counts():
    y = np.array([1, 1, 0, 0, 1]); s = np.array([0.9, 0.2, 0.8, 0.1, 0.7])
    m = bl.metrics_at_threshold(y, s, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 1)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)


def test_bootstrap_is_deterministic_and_ordered():
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.1).astype(int); s = y * 1.0 + rng.normal(0, 0.5, 300)
    a = bl.bootstrap_ci(y, s, 0.5, n_boot=100, seed=1); b = bl.bootstrap_ci(y, s, 0.5, n_boot=100, seed=1)
    assert a == b and all(lo <= hi for lo, hi in a.values())


def test_tfidf_is_fit_on_train_only():
    pipe = bl.build_pipeline(1.0).fit(["alpha beta", "alpha beta gamma", "alpha beta"], [0, 1, 0])
    assert "valonlytoken" not in pipe.named_steps["tfidf"].vocabulary_
    pipe.decision_function(["valonlytoken alpha"])  # unseen tokens are ignored, not learned
    assert "valonlytoken" not in pipe.named_steps["tfidf"].vocabulary_


def test_end_to_end_never_loads_test_and_writes_clean_artifacts(tmp_path, monkeypatch):
    proc = tmp_path / "processed"
    ds.run_pipeline(_signal_csv(tmp_path), proc)
    calls = []
    real = ds.load_split
    monkeypatch.setattr(ds, "load_split", lambda name, *a, **k: calls.append((name, k)) or real(name, *a, **k))
    rec = bl.run_baseline(proc, tmp_path / "rep", tmp_path / "models", c_grid=(0.3, 3.0), n_boot=30)
    assert sorted(n for n, _ in calls) == ["train", "val"]            # test split never requested
    assert rec["test_split_loaded"] is False and rec["evaluated_split"] == "val"
    assert rec["val"]["average_precision"] > 0.8                      # planted signal is learnable
    saved = (tmp_path / "rep" / "baseline_run.json").read_text()
    assert "specialist" not in saved and "western union" not in saved  # no posting text in the record
    assert json.loads(saved)["split_manifest_sha256"] == ds.file_sha256(proc / "split_manifest.json")
    assert (tmp_path / "models" / bl.MODEL_FILE).exists()
    import joblib
    pipe = joblib.load(tmp_path / "models" / bl.MODEL_FILE)["pipeline"]
    Xva, yva = bl.xy(ds.load_split("val", proc))
    assert rec["frozen_threshold"] == bl.select_threshold(yva, pipe.decision_function(Xva), "max_f1")  # from val only


def test_refuses_when_manifest_records_failed_leakage_or_wrong_threshold(tmp_path):
    proc = tmp_path / "processed"
    ds.run_pipeline(_signal_csv(tmp_path), proc)
    mp = proc / "split_manifest.json"
    m = json.loads(mp.read_text())
    m["leakage"]["all_passed"] = False
    mp.write_text(json.dumps(m))
    with pytest.raises(RuntimeError, match="leakage"):
        bl.run_baseline(proc, tmp_path / "r", tmp_path / "m", c_grid=(1.0,), n_boot=5)
    m["leakage"]["all_passed"] = True; m["near_duplicate_threshold"] = 0.95
    mp.write_text(json.dumps(m))
    with pytest.raises(RuntimeError, match="0.90"):
        bl.run_baseline(proc, tmp_path / "r", tmp_path / "m", c_grid=(1.0,), n_boot=5)


def test_append_results_schema(tmp_path):
    p = tmp_path / "results.csv"
    p.write_text("person,experiment,model,metric,value,date\n")
    bl.append_results(p, "me", "baseline_val", "tfidf_lr", {"val_fraud_recall": 0.8123456789})
    df = pd.read_csv(p)
    assert list(df.columns) == bl.RESULTS_COLUMNS and df.loc[0, "value"] == 0.812346
