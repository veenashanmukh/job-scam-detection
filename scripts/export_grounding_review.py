"""Export a human-review sample of the LLM grounding run. Read-only: no LLM, no network, never opens test.csv.

Input: reports/grounding_runs/grounding_val_openai_gpt-oss-120b.jsonl (the finished validation run).
Output: reports/grounding_runs/review_sample.md (gitignored folder: it quotes validation posting text) holding
  * every record whose llm_status is not "ok" (the attributes_reasoning rejections), and
  * 20 records with status "ok", chosen at random with seed 42.
For each record: posting id, true label, verdict and display score, the posting text (first 1,500 characters), the
evidence items with their ids and text, and the generated explanation with its citations. A blank review table
follows the sample.

What the run file does NOT contain, and how it is rebuilt here (offline):
  * posting text and fields: from data/processed_v2/val.csv by job_id;
  * display score: from the saved validation log-odds (val_probabilities.npy), same formula as the app;
  * evidence items: rebuilt with the current retrieval caches and rules (posting index, entity graph, local
    guidance index). The run stored only the cited ids and the number of ids, so every record says whether the
    rebuilt id list lines up with the run; if it does not, the rules or caches changed since the run and the ids
    cannot be trusted for that record.
  * for records the checker rejected, the run stored the template fallback, not the rejected LLM text. Reviewing the
    rejected wording itself would need a new LLM call, which this script never makes.

Usage: python scripts/export_grounding_review.py [--runs PATH] [--out PATH] [--n-ok 20]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services import dataset as ds  # noqa: E402
from backend.services.evidence import build_evidence, evidence_ids  # noqa: E402
from backend.services.job_analysis import (  # noqa: E402
    detect_warning_signs, display_risk_score, escalate_verdict, job_verdict,
)

SEED = 42
DEFAULT_N_OK = 20
POSTING_CHARS = 1500
THRESHOLD = 0.291163
DEFAULT_RUNS = ROOT / "reports" / "grounding_runs" / "grounding_val_openai_gpt-oss-120b.jsonl"
DEFAULT_OUT = ROOT / "reports" / "grounding_runs" / "review_sample.md"
VAL_CSV = ROOT / "data" / "processed_v2" / "val.csv"
VAL_PROBABILITIES = ROOT / "backend" / "models_v2" / "distilbert_job_scam" / "val_probabilities.npy"
TABLE_HEADER = ("| id | each cited claim supported by its evidence? (yes / partly / no) "
                "| contradicts the verdict? | notes |")


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_records(records: list[dict], n_ok: int = DEFAULT_N_OK) -> list[dict]:
    """Every non-ok record, then n_ok ok records chosen at random (seed 42); each group in job_id order."""
    not_ok = sorted((r for r in records if r["llm_status"] != "ok"), key=lambda r: int(r["job_id"]))
    ok = sorted((r for r in records if r["llm_status"] == "ok"), key=lambda r: int(r["job_id"]))
    chosen = np.random.default_rng(SEED).choice(len(ok), min(n_ok, len(ok)), replace=False) if ok else []
    return not_ok + [ok[i] for i in sorted(chosen)]


def trim(text: str, limit: int = POSTING_CHARS) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + f" [trimmed to {limit} characters]"


def _one_line(text) -> str:
    return " ".join(str(text or "").split())


def _quote(text: str) -> list[str]:
    return [("> " + line) if line.strip() else ">" for line in text.splitlines()] or [">"]


def evidence_lines(evidence: dict, cited: set) -> list[str]:
    """One bullet per evidence item with its id and text; items the explanation cited are marked."""
    lines = []

    def add(item_id, body):
        lines.append(f"- **[{item_id}]**{' (cited)' if item_id in cited else ''} {body}")

    for s in evidence.get("similar_postings", []):
        add(s["id"], f"similar known posting, {'fraudulent' if s['label'] == 1 else 'legitimate'}, "
                     f"similarity {s['similarity']:.2f}: {_one_line(s['snippet'])}")
    for g in evidence.get("graph", {}).get("items", []):
        add(g["id"], g.get("message") or f"This {g['kind_name']} appears in {g['linked_count']} known postings "
                                         f"({g['fraud_count']} labelled fraud).")
    for k in evidence.get("guidance", []):
        add(k["id"], f"guidance, {k.get('publisher', '')} - {k.get('title', '')}: {_one_line(k.get('excerpt'))}")
    for r in evidence.get("rule_signs", []):
        add(r["id"], f"rule-based warning sign {r['label']}: matched \"{_one_line(r['matched_text'])}\"")
    for t in evidence.get("messages", []):
        add(t["id"], f"flagged message: {_one_line(t['text'])}")
    return lines or ["- (no evidence items)"]


def posting_fields(row: dict) -> dict:
    return {f: (None if pd.isna(row.get(f)) else row[f]) for f in ds.TEXT_FIELDS}


def record_section(number: int, record: dict, row: dict, logodds: float, evidence: dict) -> list[str]:
    fields = posting_fields(row)
    text = ds.build_text(pd.Series(fields))
    profile = ds.clean_field(fields["company_profile"]) or ""
    score = display_risk_score(logodds, THRESHOLD)
    ids = evidence_ids(evidence)
    cited = list(record.get("cited_evidence_ids") or [])
    ok = record["llm_status"] == "ok"
    label = "fraud" if int(record["fraudulent"]) == 1 else "legitimate"

    lines = [f"## {number}. Posting {record['job_id']} - true label: {label}",
             "",
             f"- Verdict (as in the run): **{record['verdict']}**; display score: **{score}** "
             "(from the saved validation log-odds)",
             f"- Run status: {record['llm_status']} / {record['llm_detail']}; explanation source: "
             f"{record['explanation_source']}"]

    current = job_verdict(1 / (1 + np.exp(-logodds)), THRESHOLD, profile)
    current_verdict = escalate_verdict(current[0], current[1], detect_warning_signs(text))[0]
    if current_verdict != record["verdict"]:
        lines.append(f"- NOTE: the current code would give **{current_verdict}** for this posting; the verdict above "
                     "is the one the explanation was written for.")
    problems = []
    if len(ids) != int(record.get("n_evidence_ids", len(ids))):
        problems.append(f"the rebuilt evidence has {len(ids)} items but the run had {record.get('n_evidence_ids')}")
    missing = [i for i in cited if i not in ids]
    if missing:
        problems.append(f"cited ids {missing} do not exist in the rebuilt evidence")
    if problems:
        lines.append("- WARNING: evidence does not line up with the run (" + "; ".join(problems) + "). The rules or "
                     "caches changed since the run, so the evidence below may differ from what the model saw.")

    lines += ["", f"**Posting text** (first {POSTING_CHARS:,} characters)", ""]
    lines += _quote(trim(text)) + ["", "**Evidence items**", ""] + evidence_lines(evidence, set(cited)) + [""]
    if ok:
        lines.append(f"**Generated explanation** (LLM; cited: {', '.join(cited) if cited else 'none'})")
    else:
        lines.append(f"**Stored explanation** (template fallback. The LLM answer was rejected as "
                     f"{record['llm_detail']} and the rejected wording was not saved in the run; "
                     f"template cites: {', '.join(cited) if cited else 'none'})")
    return lines + [""] + _quote(str(record["explanation"])) + [""]


def review_table(records: list[dict]) -> list[str]:
    return ["## Review table", "", "Fill in one row per posting. For each cited claim, check it against the evidence "
            "item with the same id.", "", TABLE_HEADER, "|---|---|---|---|"] + [
        f"| {r['job_id']} | | | |" for r in records]


def build_review(selected: list[dict], val: pd.DataFrame, probabilities: np.ndarray, evidence_for) -> str:
    """evidence_for(text, fields, signs) -> evidence dict (the real build_evidence call, or a fake in tests)."""
    index = {int(job_id): i for i, job_id in enumerate(val[ds.ID_COL])}
    n_not_ok = sum(1 for r in selected if r["llm_status"] != "ok")
    lines = ["# Grounding review sample", "",
             f"{len(selected)} records from the validation run: {n_not_ok} not ok (all of them) and "
             f"{len(selected) - n_not_ok} ok chosen at random (seed {SEED}). Validation postings only. "
             "Contains posting text: do not commit or share.", ""]
    for number, record in enumerate(selected, 1):
        position = index[int(record["job_id"])]
        row = val.iloc[position].to_dict()
        p = float(np.clip(probabilities[position], 1e-12, 1 - 1e-12))
        logodds = float(np.log(p / (1 - p)))
        fields = posting_fields(row)
        text = ds.build_text(pd.Series(fields))
        evidence = evidence_for(text, fields, detect_warning_signs(text))
        lines += record_section(number, record, row, logodds, evidence)
    return "\n".join(lines + review_table(selected)) + "\n"


def make_real_evidence_builder():
    """Offline retrieval only: local caches and the local guidance index (no model downloads, no LLM)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from backend.services import entity_graph, posting_index

    train_sha = posting_index.manifest_train_sha256()
    resources = {}
    for key, load in (("posting_index", posting_index.load_index), ("graph", entity_graph.load_graph)):
        try:
            resources[key] = load(train_sha)
        except Exception:
            resources[key] = None

    def guidance(text):
        from backend.services.semantic_retrieval import retrieve_semantic

        return retrieve_semantic(text, top_k=3, local_files_only=True)

    def evidence_for(text, fields, signs):
        return build_evidence(text, fields, signs, posting_index=resources["posting_index"],
                              graph=resources["graph"], guidance_fn=guidance)

    return evidence_for


def main(argv=None, evidence_for=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=DEFAULT_RUNS, help="grounding run results (jsonl)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="markdown file to write")
    ap.add_argument("--n-ok", type=int, default=DEFAULT_N_OK, help="ok records to sample (default 20)")
    ap.add_argument("--val", type=Path, default=VAL_CSV, help=argparse.SUPPRESS)
    ap.add_argument("--probabilities", type=Path, default=VAL_PROBABILITIES, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if not args.runs.exists():
        print(f"Run file not found: {args.runs.name}", file=sys.stderr)
        return 2
    records = load_records(args.runs)
    selected = select_records(records, args.n_ok)
    val = pd.read_csv(args.val)
    probabilities = np.load(args.probabilities)
    if len(val) != len(probabilities):
        print("val.csv and val_probabilities.npy have different lengths.", file=sys.stderr)
        return 3
    unknown = sorted({int(r["job_id"]) for r in selected} - set(int(j) for j in val[ds.ID_COL]))
    if unknown:
        print(f"{len(unknown)} run records are not in val.csv (first: {unknown[0]}); wrong run file?", file=sys.stderr)
        return 3
    text = build_review(selected, val, probabilities, evidence_for or make_real_evidence_builder())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    n_not_ok = sum(1 for r in selected if r["llm_status"] != "ok")
    print(f"Wrote {len(selected)} records ({n_not_ok} not ok, {len(selected) - n_not_ok} ok sampled with seed {SEED}) "
          f"to {args.out.name}; warnings about misaligned evidence: {text.count('- WARNING:')}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
