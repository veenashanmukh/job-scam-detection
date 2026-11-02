import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import eval_chats_synthetic as ev  # noqa: E402

SCAM = ("24/10/26, 9:00 - Recruiter: Pay the registration fee of Rs 3000 within 2 hours.\n"
        "24/10/26, 9:01 - Recruiter: Send your bank details and add me on Telegram.\n"
        "24/10/26, 9:02 - You: Why?\n")
BENIGN = "24/10/26, 9:00 - HR: Thanks for applying.\n24/10/26, 9:01 - You: When is my interview?\n"


def _write(tmp_path, entries):
    (tmp_path / "chats").mkdir()
    labels = []
    for i, (label, text) in enumerate(entries):
        (tmp_path / "chats" / f"c{i}.txt").write_text(text, encoding="utf-8")
        labels.append({"id": f"c{i}", "file": f"chats/c{i}.txt", "label": label, "difficulty": "easy",
                       "type": "t", "you": "You"})
    (tmp_path / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    return labels


def test_scoring_metrics_and_report(tmp_path):
    labels = _write(tmp_path, [("scam", SCAM), ("scam", SCAM), ("benign", BENIGN), ("benign", BENIGN)])
    rows = ev.score_chats(labels, tmp_path)
    assert [r["category"] for r in rows][2:] == ["Lower risk", "Lower risk"]
    m = ev.metrics(rows, ev.CUTOFFS["Higher+Moderate"])
    assert m["accuracy"] == 1.0 and m["recall"] == 1.0 and m["fp"] == 0
    text = "\n".join(ev.report_lines(rows))
    assert ev.BANNER in text and "Misclassified at cutoff Higher only: 0" in text


def test_metrics_count_a_missed_scam():
    rows = [{"label": "scam", "category": "Lower risk"}, {"label": "benign", "category": "Lower risk"}]
    m = ev.metrics(rows, ev.CUTOFFS["Higher only"])
    assert m["fn"] == 1 and m["recall"] == 0.0 and m["f1"] == 0.0


def test_wrong_size_labels_exit_with_error(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(ev, "DEFAULT_DIR", tmp_path)  # the exact 16/16 check applies to the default directory
    _write(tmp_path, [("scam", SCAM), ("benign", BENIGN)] * 2)
    with pytest.raises(ValueError):
        ev.load_labels(tmp_path)
    assert ev.main(tmp_path) == 2
    assert "exactly 32" in capsys.readouterr().err


def test_data_dir_with_two_chats_and_json_out(tmp_path, capsys, monkeypatch):
    _write(tmp_path, [("scam", SCAM), ("benign", BENIGN)])
    out = tmp_path / "res.json"
    monkeypatch.setattr(sys, "argv", ["eval_chats_synthetic.py", "--data-dir", str(tmp_path), "--json-out", str(out)])
    assert ev.main(tmp_path, out) == 0
    printed = capsys.readouterr().out
    assert str(tmp_path) in printed and "1 scam, 1 benign" in printed and ev.BANNER in printed
    assert len(json.loads(out.read_text(encoding="utf-8"))["chats"]) == 2


def test_custom_dir_needs_both_labels(tmp_path):
    _write(tmp_path, [("scam", SCAM), ("scam", SCAM)])
    with pytest.raises(ValueError):
        ev.load_labels(tmp_path)
