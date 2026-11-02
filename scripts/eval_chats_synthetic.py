"""Score the synthetic labelled chats with the rule-based chat analysis. Offline: no LLM, no evidence step.

Reads data/synthetic_chats/labels.json (32 chats: 16 scam, 16 benign), parses each chat and scores it
with analyze_messages using the "you" name from the labels. Two cutoffs are reported: scam = "Higher risk"
only, and scam = "Higher risk" or "Moderate risk".

Usage: python scripts/eval_chats_synthetic.py [--data-dir DIR] [--json-out PATH]

The exact 16/16 check applies to the default directory only; any other directory needs at least one
scam and one benign chat.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services.conversation_analysis import analyze_messages  # noqa: E402
from backend.services.conversation_parser import parse_chat_text  # noqa: E402

DEFAULT_DIR = ROOT / "data" / "synthetic_chats"
EXPECTED = {"scam": 16, "benign": 16}
CUTOFFS = {"Higher only": {"Higher risk"}, "Higher+Moderate": {"Higher risk", "Moderate risk"}}
BANNER = "SYNTHETIC SET - BEHAVIOUR DEMONSTRATION, NOT REAL-WORLD ACCURACY"


def load_labels(directory: Path) -> list[dict]:
    labels = json.loads((directory / "labels.json").read_text(encoding="utf-8"))
    counts = {k: sum(1 for c in labels if c["label"] == k) for k in EXPECTED}
    if directory.resolve() == DEFAULT_DIR.resolve():
        if len(labels) != 32 or counts != EXPECTED:
            raise ValueError(f"labels.json must hold exactly 32 chats (16 scam, 16 benign); "
                             f"found {len(labels)} ({counts}).")
    elif not counts["scam"] or not counts["benign"]:
        raise ValueError(f"labels.json needs at least 1 scam and 1 benign chat; found {counts}.")
    return labels


def score_chats(labels: list[dict], directory: Path) -> list[dict]:
    rows = []
    for item in labels:
        text = (directory / item["file"]).read_text(encoding="utf-8-sig")
        report = analyze_messages(parse_chat_text(text), self_name=item.get("you") or None)
        rows.append({**item, "risk_score": report["risk_score"], "category": report["risk_category"],
                     "signs": report["warning_signs"]})
    return rows


def metrics(rows: list[dict], scam_categories: set) -> dict:
    tp = sum(1 for r in rows if r["label"] == "scam" and r["category"] in scam_categories)
    fn = sum(1 for r in rows if r["label"] == "scam" and r["category"] not in scam_categories)
    fp = sum(1 for r in rows if r["label"] != "scam" and r["category"] in scam_categories)
    tn = len(rows) - tp - fn - fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": (tp + tn) / len(rows), "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def correct(row: dict, scam_categories: set) -> bool:
    return (row["category"] in scam_categories) == (row["label"] == "scam")


def report_lines(rows: list[dict], directory: Path = DEFAULT_DIR) -> list[str]:
    scam = sum(1 for r in rows if r["label"] == "scam")
    out = ["=" * 70, BANNER, "=" * 70, f"Data directory: {directory}",
           f"{len(rows)} chats ({scam} scam, {len(rows) - scam} benign), rule-based scoring only (no LLM).", ""]
    for name, cats in CUTOFFS.items():
        m = metrics(rows, cats)
        out.append(f"Cutoff {name}: accuracy {m['accuracy']:.3f} precision {m['precision']:.3f} "
                   f"recall {m['recall']:.3f} F1 {m['f1']:.3f}  (TP {m['tp']} FP {m['fp']} TN {m['tn']} FN {m['fn']})")
    for title, key in (("difficulty", "difficulty"), ("label", "label")):
        out += ["", f"By {title} (share classified correctly):"]
        groups = defaultdict(list)
        for r in rows:
            groups[r[key]].append(r)
        for value in sorted(groups, key=lambda v: ["easy", "medium", "hard"].index(v) if v in ("easy", "medium", "hard") else v):
            g = groups[value]
            cells = ", ".join(f"{n}: {sum(correct(r, c) for r in g)}/{len(g)}" for n, c in CUTOFFS.items())
            out.append(f"  {value:<8} n={len(g):<3} {cells}")
    for name, cats in CUTOFFS.items():
        wrong = [r for r in rows if not correct(r, cats)]
        out += ["", f"Misclassified at cutoff {name}: {len(wrong)}"]
        for r in wrong:
            out.append(f"  {r['id']} type={r['type']} label={r['label']} difficulty={r['difficulty']} "
                       f"risk_score={r['risk_score']} category={r['category']} signs={r['signs'] or 'none'}")
    return out


def main(directory: Path = DEFAULT_DIR, json_out: Path | None = None) -> int:
    try:
        labels = load_labels(directory)
        rows = score_chats(labels, directory)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # the data path may hold characters the console cannot show
    print("\n".join(report_lines(rows, directory)))
    if json_out is not None:
        json_out.write_text(json.dumps({"data_dir": str(directory),
                                        "cutoffs": {n: metrics(rows, c) for n, c in CUTOFFS.items()},
                                        "chats": rows}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DIR, help="folder with labels.json and chats/")
    ap.add_argument("--json-out", type=Path, default=None, help="optional path for the results as JSON")
    cli = ap.parse_args()
    sys.exit(main(cli.data_dir, cli.json_out))
