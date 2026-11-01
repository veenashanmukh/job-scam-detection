"""The scripts that can read the locked test split refuse to parse it without an explicit flag.

Synthetic files in tmp_path only; the real test split is never touched.
"""
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from backend.services import dataset as ds

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def read_log(monkeypatch):
    """Record every pandas.read_csv path while still reading the file."""
    seen = []
    original = pd.read_csv

    def recording_read_csv(path, *args, **kwargs):
        seen.append(Path(path).name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", recording_read_csv)
    return seen


# --- evaluate_baseline_test.py ------------------------------------------------

class _FakePipeline:
    def decision_function(self, texts):
        return np.array([1.0 if "fee" in t else -1.0 for t in texts])


@pytest.fixture()
def baseline_eval(monkeypatch, tmp_path):
    import evaluate_baseline_test as ebt

    test_csv = tmp_path / "test.csv"
    test_csv.write_text("text,fraudulent\npay a fee,1\nsoftware role,0\nfee today,1\nnurse,0\n", encoding="utf-8")
    monkeypatch.setattr(ebt, "TEST_PATH", test_csv)
    monkeypatch.setattr(ebt, "OUTPUT_PATH", tmp_path / "reports" / "baseline_test_results.json")
    monkeypatch.setattr(ebt, "MODEL_PATH", tmp_path / "model.joblib")

    loads = []

    def fake_load(path):
        loads.append(path)
        return {"pipeline": _FakePipeline(), "threshold": 0.0}

    monkeypatch.setattr(joblib, "load", fake_load)
    return ebt, loads


def test_baseline_eval_refuses_without_final_evaluation_flag(baseline_eval, read_log, capsys):
    ebt, loads = baseline_eval
    assert ebt.main([]) == ebt.EXIT_NO_FLAG != 0
    assert read_log == [] and loads == []
    assert not ebt.OUTPUT_PATH.exists()
    assert "Refusing" in capsys.readouterr().err


def test_baseline_eval_refuses_to_overwrite_existing_results(baseline_eval, read_log, capsys):
    ebt, loads = baseline_eval
    ebt.OUTPUT_PATH.parent.mkdir(parents=True)
    ebt.OUTPUT_PATH.write_text('{"kept": true}', encoding="utf-8")

    assert ebt.main(["--final-evaluation"]) == ebt.EXIT_RESULTS_EXIST != 0
    assert read_log == [] and loads == []
    assert json.loads(ebt.OUTPUT_PATH.read_text(encoding="utf-8")) == {"kept": True}
    assert "--overwrite" in capsys.readouterr().err


def test_baseline_eval_runs_only_with_flags(baseline_eval, read_log):
    ebt, loads = baseline_eval
    assert ebt.main(["--final-evaluation"]) == 0
    assert read_log == ["test.csv"] and len(loads) == 1
    first = json.loads(ebt.OUTPUT_PATH.read_text(encoding="utf-8"))
    assert first["n_test"] == 4 and first["recall"] == 1.0

    assert ebt.main(["--final-evaluation", "--overwrite"]) == 0


def test_importing_baseline_eval_has_no_side_effects(read_log):
    import importlib

    import evaluate_baseline_test as ebt
    importlib.reload(ebt)
    assert read_log == []


# --- scripts/verify_splits.py -------------------------------------------------

def _load_verify_module():
    spec = importlib.util.spec_from_file_location("verify_splits", ROOT / "scripts" / "verify_splits.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def synthetic_splits(tmp_path_factory):
    from backend.tests.conftest import make_fixture

    base = tmp_path_factory.mktemp("splits")
    raw = base / "fake_job_postings.csv"
    make_fixture().to_csv(raw, index=False)
    out = base / "cand"
    ds.run_pipeline(raw, out)
    return raw, out


def _verify(monkeypatch, raw, out, *extra):
    monkeypatch.setattr(sys, "argv", ["verify_splits.py", "--dir", str(out), "--raw", str(raw), *extra])
    return _load_verify_module().main()


def test_verify_splits_hashes_but_does_not_parse_test_by_default(synthetic_splits, monkeypatch, read_log, capsys):
    raw, out = synthetic_splits
    assert _verify(monkeypatch, raw, out) == 0
    stdout = capsys.readouterr().out
    assert "test.csv" not in read_log and {"train.csv", "val.csv"} <= set(read_log)
    assert "[PASS] split file hashes match manifest" in stdout
    assert stdout.count("[SKIP]") == 3 and "--include-test" in stdout


def test_verify_splits_still_detects_tampered_test_file_without_parsing_it(synthetic_splits, monkeypatch, read_log, tmp_path):
    import shutil

    raw, out = synthetic_splits
    tampered = tmp_path / "tampered"
    shutil.copytree(out, tampered)
    with open(tampered / "test.csv", "a", encoding="utf-8") as f:
        f.write("\n")

    assert _verify(monkeypatch, raw, tampered) == 5
    assert "test.csv" not in read_log


def test_verify_splits_parses_test_only_with_include_test(synthetic_splits, monkeypatch, read_log, capsys):
    raw, out = synthetic_splits
    assert _verify(monkeypatch, raw, out, "--include-test") == 0
    stdout = capsys.readouterr().out
    assert "test.csv" in read_log
    assert "[SKIP]" not in stdout and "[PASS] recomputed: no_cross_split_near_duplicates" in stdout
