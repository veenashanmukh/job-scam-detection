"""final_test_eval on a tiny synthetic fixture in a temp directory. Never touches the real data or models."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import final_test_eval as fe  # noqa: E402
from backend.services import dataset as ds  # noqa: E402
from backend.services.baseline import build_pipeline  # noqa: E402

FRAUD_WORDS = ["pay a fee now", "send registration fee today", "earn cash fast no interview", "deposit required to start"]
LEGIT_WORDS = ["software team reviews code", "engineer builds features", "nurse supports patients", "analyst writes reports"]


def fake_logodds(texts):
    return np.array([(2.5 if ("fee" in t or "cash" in t or "deposit" in t) else -2.5) + (len(t) % 5) * 0.2 - 0.4
                     for t in texts])


def _frame(first_id, n, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        fraud = int(i % 3 == 0)
        words = (FRAUD_WORDS if fraud else LEGIT_WORDS)[int(rng.integers(0, 4))]
        if rng.random() < 0.15:  # a few confusing postings so both classes land on both sides of the thresholds
            words = (LEGIT_WORDS if fraud else FRAUD_WORDS)[int(rng.integers(0, 4))]
        rows.append({"job_id": first_id + i, "group_id": first_id + i // 2, "fraudulent": fraud,
                     "company_profile": "" if i % 2 else "Some Firm", "text": f"Description: {words} item{first_id + i}"})
    return pd.DataFrame(rows)


@pytest.fixture()
def fixture(tmp_path):
    processed = tmp_path / "processed"
    model_dir = tmp_path / "distilbert"
    processed.mkdir()
    model_dir.mkdir()
    frames = {"train": _frame(1000, 90, 1), "val": _frame(2000, 60, 2), "test": _frame(3000, 60, 3)}
    for name, frame in frames.items():
        frame.to_csv(processed / f"{name}.csv", index=False)
    manifest = {"splits": {n: {"file": f"{n}.csv", "sha256": ds.file_sha256(processed / f"{n}.csv")} for n in frames}}
    (processed / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    (model_dir / "model.safetensors").write_bytes(b"fake weights")
    val_texts = frames["val"]["text"].tolist()
    np.save(model_dir / "val_probabilities.npy", fe.sigmoid(fake_logodds(val_texts)))
    np.save(model_dir / "val_labels.npy", frames["val"]["fraudulent"].to_numpy())

    pipe = build_pipeline(30.0).fit(frames["train"]["text"], frames["train"]["fraudulent"])
    baseline_path = tmp_path / "baseline.joblib"
    joblib.dump({"pipeline": pipe, "threshold": -1.516}, baseline_path)

    test_hash = manifest["splits"]["test"]["sha256"]
    weights_hash = ds.file_sha256(model_dir / "model.safetensors")
    cfg = fe.Config(processed_dir=processed, model_dir=model_dir, baseline_path=baseline_path,
                    out_dir=tmp_path / "out", test_hash_fragments=(test_hash[:8], test_hash[-14:]),
                    weights_hash_fragments=(weights_hash[:8], weights_hash[-8:]), n_boot=40, cv_folds=3)
    return cfg, frames


class Factory:
    """Records whether a scorer was ever requested, i.e. whether the run got past the integrity checks."""

    def __init__(self, shift=0.0):
        self.calls, self.shift = 0, shift

    def __call__(self):
        self.calls += 1
        return lambda texts: fake_logodds(texts) + self.shift


@pytest.fixture(autouse=True)
def no_git(monkeypatch):
    monkeypatch.setattr(fe, "git_info", lambda: {"head_commit": "abc123", "script_status": "untracked"})


# --- refusal paths ---------------------------------------------------------------------

def test_refuses_without_the_flag_and_prints_the_banner(tmp_path, capsys):
    cfg = fe.Config(processed_dir=tmp_path / "missing", out_dir=tmp_path / "out")  # no data exists at all
    factory = Factory()
    assert fe.main([], cfg, factory) == 2
    captured = capsys.readouterr()
    assert fe.BANNER in captured.out and "second look at the test split" in captured.out
    assert "reports/distilbert_v1/test_results.json" in captured.out
    assert "--final-evaluation" in captured.err and factory.calls == 0


def test_there_is_no_overwrite_option(fixture):
    cfg, _ = fixture
    with pytest.raises(SystemExit) as exc:
        fe.main(["--final-evaluation", "--overwrite"], cfg, Factory())
    assert exc.value.code == 2


def test_refuses_when_results_exist_without_touching_anything(fixture, capsys):
    cfg, _ = fixture
    cfg.out_dir.mkdir()
    cfg.results_path.write_text('{"old": true}', encoding="utf-8")
    factory = Factory()
    assert fe.main(["--final-evaluation"], cfg, factory) == 2
    assert cfg.results_path.read_text(encoding="utf-8") == '{"old": true}' and factory.calls == 0
    assert "already exists" in capsys.readouterr().err


# --- integrity stops (nothing is scored, nothing is written) ---------------------------------

def _stopped(cfg, factory, capsys, message):
    assert fe.main(["--final-evaluation"], cfg, factory) == 3
    assert message in capsys.readouterr().err
    assert factory.calls == 0 and not cfg.results_path.exists() and not cfg.summary_path.exists()


def test_test_csv_hash_mismatch_stops(fixture, capsys):
    cfg, frames = fixture
    with (cfg.processed_dir / "test.csv").open("a", encoding="utf-8") as f:
        f.write("9999,9999,1,,Description: tampered\n")
    _stopped(cfg, Factory(), capsys, "test split does not match the manifest")


def test_train_hash_mismatch_stops(fixture, capsys):
    cfg, _ = fixture
    (cfg.processed_dir / "train.csv").write_text("job_id\n1\n", encoding="utf-8")
    _stopped(cfg, Factory(), capsys, "train split does not match the manifest")


def test_unexpected_test_hash_fragment_stops(fixture, capsys):
    cfg, _ = fixture
    cfg.test_hash_fragments = ("00000000", cfg.test_hash_fragments[1])
    _stopped(cfg, Factory(), capsys, "test.csv hash does not match the expected locked hash")


def test_weights_hash_mismatch_stops(fixture, capsys):
    cfg, _ = fixture
    (cfg.model_dir / "model.safetensors").write_bytes(b"different weights")
    _stopped(cfg, Factory(), capsys, "weights hash does not match")


def test_changed_baseline_threshold_stops(fixture, capsys):
    cfg, _ = fixture
    artifact = joblib.load(cfg.baseline_path)
    joblib.dump({"pipeline": artifact["pipeline"], "threshold": -0.5}, cfg.baseline_path)
    _stopped(cfg, Factory(), capsys, "baseline threshold")


def test_scorer_that_does_not_reproduce_validation_stops(fixture, capsys):
    cfg, _ = fixture
    factory = Factory(shift=0.5)
    assert fe.main(["--final-evaluation"], cfg, factory) == 3
    assert "does not reproduce" in capsys.readouterr().err and not cfg.results_path.exists()


def test_test_split_overlapping_train_stops(fixture, capsys):
    cfg, frames = fixture
    bad = frames["test"].copy()
    bad.loc[0, "job_id"] = frames["train"].loc[0, "job_id"]
    bad.to_csv(cfg.processed_dir / "test.csv", index=False)
    manifest = json.loads((cfg.processed_dir / "split_manifest.json").read_text(encoding="utf-8"))
    manifest["splits"]["test"]["sha256"] = ds.file_sha256(cfg.processed_dir / "test.csv")
    (cfg.processed_dir / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    sha = manifest["splits"]["test"]["sha256"]
    cfg.test_hash_fragments = (sha[:8], sha[-14:])
    assert fe.main(["--final-evaluation"], cfg, Factory()) == 3
    assert "overlaps" in capsys.readouterr().err and not cfg.results_path.exists()


# --- a complete run on the fixture ------------------------------------------------------------

def test_full_run_writes_the_expected_schema_and_cannot_be_repeated(fixture, capsys):
    cfg, frames = fixture
    assert fe.main(["--final-evaluation"], cfg, Factory()) == 0
    result = json.loads(cfg.results_path.read_text(encoding="utf-8"))
    summary = cfg.summary_path.read_text(encoding="utf-8")

    assert result["banner"] == fe.BANNER and result["git"] == {"head_commit": "abc123", "script_status": "untracked"}
    assert result["timestamp_utc"].endswith("+00:00")
    assert set(result["hashes"]) == {"train_csv_sha256", "val_csv_sha256", "test_csv_sha256",
                                     "distilbert_weights_sha256", "baseline_joblib_sha256"}
    assert result["data"]["test_rows"] == 60 and result["data"]["test_fraud"] == int(frames["test"]["fraudulent"].sum())
    assert result["data"]["bootstrap"]["unit"] == "group_id" and result["data"]["bootstrap"]["seed"] == 42
    assert result["frozen_choices"]["distilbert_threshold"] == 0.291163 and result["frozen_choices"]["baseline_C"] == 30.0
    assert result["frozen_choices"]["combiner"]["fitted_on"] == "validation only"

    assert set(result["models"]) == {"distilbert", "baseline", "combiner_exploratory"}
    assert result["models"]["combiner_exploratory"]["label"] == "exploratory, frozen on validation"
    for block in result["models"].values():
        for metric in ("pr_auc", "roc_auc", "precision", "recall", "f1"):
            assert set(block[metric]) == {"value", "ci95"} and block[metric]["ci95"][0] <= block[metric]["ci95"][1]
        assert sum(block["confusion"].values()) == 60
    assert result["models"]["distilbert"]["threshold"] == 0.291163

    diff = result["paired_pr_auc_difference_distilbert_minus_baseline"]
    assert set(diff) == {"value", "ci95", "share_of_resamples_above_zero"}
    assert [t["tier"] for t in result["verdict_tiers"]] == list(fe.TIERS)
    assert sum(t["n"] for t in result["verdict_tiers"]) == 60
    assert all(0.0 <= t["fraud_rate_ci95_clopper_pearson"][0] <= t["fraud_rate_ci95_clopper_pearson"][1] <= 1.0
               for t in result["verdict_tiers"])
    cal = result["calibration"]
    assert cal["temperature"] > 0 and cal["temperature_fitted_on"] == "validation only"
    assert set(cal) >= {"ece_before", "ece_after", "log_loss_before", "log_loss_after"}
    assert sum(b["n"] for b in result["display_score_bands"]) <= 60

    assert fe.BANNER in summary and "exploratory, frozen on validation" in summary and "Verdict tiers" in summary

    before = cfg.results_path.read_text(encoding="utf-8")
    assert fe.main(["--final-evaluation"], cfg, Factory()) == 2  # a second run is refused
    assert cfg.results_path.read_text(encoding="utf-8") == before


def test_results_file_is_created_exclusively(fixture, monkeypatch):
    cfg, _ = fixture
    monkeypatch.setattr(fe, "summary_markdown", lambda result: "summary")
    cfg.out_dir.mkdir()
    cfg.results_path.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        fe.write_outputs(cfg, {"timestamp_utc": "t", "git": {"head_commit": "x", "script_status": "y"}})


# --- maths -------------------------------------------------------------------------------------

def test_clopper_pearson_known_values():
    lo, hi = fe.clopper_pearson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.3085, abs=1e-3)
    lo, hi = fe.clopper_pearson(5, 10)
    assert (lo, hi) == pytest.approx((0.1871, 0.8129), abs=1e-3)
    assert fe.clopper_pearson(10, 10)[1] == 1.0 and np.isnan(fe.clopper_pearson(0, 0)[0])


def test_ece_and_log_loss():
    y = np.array([1, 0, 1, 0])
    assert fe.expected_calibration_error(y, y.astype(float)) == pytest.approx(0.0)
    assert fe.expected_calibration_error(y, np.full(4, 0.9)) == pytest.approx(0.4)
    assert fe.log_loss(y, np.full(4, 0.5)) == pytest.approx(np.log(2))


def test_temperature_fit_recovers_the_scale_and_uses_only_its_inputs():
    rng = np.random.default_rng(0)
    lo = rng.normal(0, 4, 20000)
    y = (rng.random(20000) < fe.sigmoid(lo / 2.0)).astype(int)
    fit = fe.fit_temperature(lo, y)
    assert fit["T"] == pytest.approx(2.0, abs=0.15) and fit["val_log_loss_after"] < fit["val_log_loss_before"]


def test_group_bootstrap_resamples_whole_groups():
    groups = np.array([0, 0, 0, 1, 2, 2])
    index = fe.group_index(groups)
    assert sorted(len(i) for i in index) == [1, 2, 3]
    y = np.array([1, 0, 1, 0, 1, 0])
    models = {n: {"score": np.arange(6.0), "pred": np.arange(6) >= 3} for n in ("distilbert", "baseline")}
    calib = {"before": np.full(6, 0.5), "after": np.full(6, 0.5)}
    out = fe.bootstrap(y, models, calib, groups, 30)
    assert out["used"] + out["skipped"] == 30 and len(out["pr_auc_diff"]) == out["used"]
    assert np.allclose(out["pr_auc_diff"], 0.0)  # identical models: paired difference is exactly zero


# --- --preflight -------------------------------------------------------------------------------

class ReadCsvRecorder:
    """Wraps pd.read_csv in the script so any attempt to read test.csv rows is caught."""

    def __init__(self, monkeypatch):
        self.paths = []
        real = pd.read_csv

        def recording(path, *args, **kwargs):
            self.paths.append(Path(path).name)
            return real(path, *args, **kwargs)

        monkeypatch.setattr(fe.pd, "read_csv", recording)


def test_preflight_runs_every_pre_test_check_without_reading_test_rows_or_writing(fixture, monkeypatch, capsys):
    cfg, frames = fixture
    recorder = ReadCsvRecorder(monkeypatch)
    scored = []

    def factory():
        return lambda texts: (scored.append(list(texts)), fake_logodds(texts))[1]

    assert fe.main(["--preflight"], cfg, factory) == 0
    out = capsys.readouterr().out
    assert fe.BANNER in out and "Preflight checks passed" in out
    for expected in ("manifest", "locked hash fragments", "baseline is C=30", "lines up", "reproduces saved validation",
                     "fitted on validation only"):
        assert expected in out
    for label in ("temperature T", "baseline threshold", "combiner threshold"):
        assert label in out
    assert "test.csv" not in recorder.paths and sorted(set(recorder.paths)) == ["train.csv", "val.csv"]
    test_texts = set(frames["test"]["text"])
    assert len(scored) == 1 and len(scored[0]) == fe.ALIGNMENT_ROWS and not test_texts & set(scored[0])
    assert not cfg.out_dir.exists()  # nothing written, not even the folder


def test_preflight_leaves_an_existing_results_file_alone_and_does_not_need_the_flag_pair(fixture):
    cfg, _ = fixture
    cfg.out_dir.mkdir()
    cfg.results_path.write_text('{"old": true}', encoding="utf-8")
    assert fe.main(["--preflight"], cfg, Factory()) == 0
    assert cfg.results_path.read_text(encoding="utf-8") == '{"old": true}'
    assert sorted(p.name for p in cfg.out_dir.iterdir()) == ["final_test_results.json"]


def test_preflight_and_final_evaluation_are_mutually_exclusive(fixture):
    cfg, _ = fixture
    with pytest.raises(SystemExit) as exc:
        fe.main(["--preflight", "--final-evaluation"], cfg, Factory())
    assert exc.value.code == 2


@pytest.mark.parametrize("break_it, message", [
    (lambda cfg: (cfg.processed_dir / "train.csv").write_text("job_id\n1\n", encoding="utf-8"), "train split"),
    (lambda cfg: (cfg.model_dir / "model.safetensors").write_bytes(b"other"), "weights hash"),
])
def test_preflight_failure_exits_3_and_writes_nothing(fixture, capsys, break_it, message):
    cfg, _ = fixture
    break_it(cfg)
    factory = Factory()
    assert fe.main(["--preflight"], cfg, factory) == 3
    err = capsys.readouterr().err
    assert "Preflight FAILED" in err and message in err
    assert factory.calls == 0 and not cfg.out_dir.exists()


def test_preflight_alignment_failure_exits_3(fixture, capsys):
    cfg, _ = fixture
    assert fe.main(["--preflight"], cfg, Factory(shift=0.5)) == 3
    assert "does not reproduce" in capsys.readouterr().err and not cfg.out_dir.exists()


# --- output ordering ---------------------------------------------------------------------------

def test_failure_while_building_the_summary_leaves_no_results_file(fixture, monkeypatch):
    cfg, _ = fixture

    def boom(_result):
        raise RuntimeError("summary failed")

    monkeypatch.setattr(fe, "summary_markdown", boom)
    with pytest.raises(RuntimeError, match="summary failed"):
        fe.write_outputs(cfg, {"timestamp_utc": "t"})
    assert not cfg.results_path.exists() and not cfg.summary_path.exists()


def test_a_full_run_whose_summary_fails_leaves_no_results_so_it_can_be_rerun(fixture, monkeypatch):
    cfg, _ = fixture
    real = fe.summary_markdown
    monkeypatch.setattr(fe, "summary_markdown", lambda r: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        fe.main(["--final-evaluation"], cfg, Factory())
    assert not cfg.results_path.exists()
    monkeypatch.setattr(fe, "summary_markdown", real)
    assert fe.main(["--final-evaluation"], cfg, Factory()) == 0 and cfg.results_path.exists()
