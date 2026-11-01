"""Grounding check of the LLM explanation on VALIDATION ONLY. Makes live LLM calls.

For up to 150 validation postings (stratified fraud/legit), runs the full pipeline
(DistilBERT -> report -> evidence -> grounded explanation) and reports citation validity,
verdict consistency, llm_status counts and the share of explanations citing evidence.

Built for free-tier rate limits:
  * no production circuit breaker; every posting gets its own LLM call, --delay seconds apart;
  * a per-minute limit waits (retry-after, else 65 s) and retries up to 2 times;
  * a per-day limit stops cleanly ("daily quota reached; rerun to continue");
  * each finished posting is appended to reports/grounding_runs/grounding_val_<model>.jsonl (gitignored)
    and skipped on the next run, so runs resume where they stopped.
Only rate-limit labels ("minute"/"day"/"other") are stored, never provider error text.
Needs GROQ_API_KEY and LLM_MODEL; refuses to run without --run-live. Never opens test.csv.

Usage: python scripts/eval_grounding_val.py --run-live [--limit 150] [--delay 20]
       e.g. a first pass on 60 postings: --run-live --limit 60
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services.llm_explainer import LLMQuotaError  # noqa: E402
from script_env import load_project_env  # noqa: E402

MAX_POSTINGS = 150
SEED = 42
DEFAULT_DELAY_SECONDS = 20.0
MINUTE_RETRIES = 2
DEFAULT_MINUTE_WAIT_SECONDS = 65.0
RUNS_DIR = ROOT / "reports" / "grounding_runs"
DAILY_MESSAGE = "daily quota reached; rerun to continue"

# Outcomes that say nothing about the model's answer; such rows are not saved and are retried next run.
RETRY_STATUSES = {"quota", "unavailable"}
RETRY_DETAILS = {"timeout", "call_failed", "not_configured", "circuit_open"}


def stratified_sample(val: pd.DataFrame, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    fraud = val.index[val["fraudulent"] == 1].to_numpy()
    legit = val.index[val["fraudulent"] == 0].to_numpy()
    n_fraud = min(len(fraud), n // 2)
    chosen = np.concatenate([rng.choice(fraud, n_fraud, replace=False),
                             rng.choice(legit, n - n_fraud, replace=False)])
    return val.loc[np.sort(chosen)]


class NoBreaker:
    """The evaluation calls the LLM for every posting; pacing replaces the circuit breaker."""

    def allow(self) -> bool:
        return True

    def record_success(self) -> None:
        pass

    def record_failure(self) -> None:
        pass


class RateLimitedClient:
    """Wrap a client: wait and retry per-minute limits; flag and re-raise per-day limits."""

    def __init__(self, client, *, sleep=time.sleep, retries=MINUTE_RETRIES,
                 default_wait=DEFAULT_MINUTE_WAIT_SECONDS):
        self.client, self.sleep, self.retries, self.default_wait = client, sleep, retries, default_wait
        self.daily_quota_reached = False

    def complete(self, system, user, timeout):
        for attempt in range(self.retries + 1):
            try:
                return self.client.complete(system, user, timeout=timeout)
            except LLMQuotaError as exc:
                if exc.kind == "day":
                    self.daily_quota_reached = True
                    raise
                if exc.kind != "minute" or attempt == self.retries:
                    raise
                self.sleep(exc.retry_after if exc.retry_after is not None else self.default_wait)
        raise AssertionError("unreachable")


def is_completed(record: dict) -> bool:
    return record["llm_status"] not in RETRY_STATUSES and record["llm_detail"] not in RETRY_DETAILS


def load_completed(path: Path) -> dict:
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                done[int(record["job_id"])] = record
    return done


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_rows(rows, analyse, out_path: Path, *, delay: float, sleep=time.sleep,
             daily_quota_reached=lambda: False, log=print) -> dict:
    """Analyse unfinished rows `delay` seconds apart, saving each completed one immediately."""
    done = load_completed(out_path)
    todo = [row for row in rows if int(row["job_id"]) not in done]
    log(f"resuming: {len(rows) - len(todo)} already completed, {len(todo)} to go")

    stopped, not_saved = False, Counter()
    for i, row in enumerate(todo):
        if i:
            sleep(delay)
        record = analyse(row)
        if daily_quota_reached():
            log(DAILY_MESSAGE)
            stopped = True
            break
        if is_completed(record):
            append_record(out_path, record)
            done[int(record["job_id"])] = record
        else:
            not_saved[record.get("rate_limit_kind") or record["llm_detail"]] += 1
        if (i + 1) % 10 == 0:
            log(f"  {i + 1}/{len(todo)} attempted this run")

    wanted = {int(row["job_id"]) for row in rows}
    completed = [record for job_id, record in done.items() if job_id in wanted]
    return {"completed": completed, "total": len(rows), "stopped_daily": stopped, "not_saved": not_saved}


def summarize(result: dict, log=print) -> None:
    """Metrics over completed rows only; always reports completed N of M."""
    records = result["completed"]
    log(f"\ncompleted {len(records)} of {result['total']}")
    if result["not_saved"]:
        log(f"not saved this run (will retry next run): {dict(result['not_saved'])}")
    if not records:
        return

    statuses = Counter(r["llm_status"] for r in records)
    details = Counter(r["llm_detail"] for r in records)
    log(f"llm_status counts: {dict(statuses)}")
    log(f"detail counts:     {dict(details)}")

    accepted_or_checked = statuses["ok"] + statuses["inconsistent"]
    answered = accepted_or_checked + details["unknown_evidence_id"]
    if answered:
        log(f"citation validity rate: {accepted_or_checked}/{answered} = {accepted_or_checked / answered:.3f} "
            "(well-formed answers whose cited ids all exist in the evidence list)")
    if accepted_or_checked:
        log(f"verdict-consistency rate: {statuses['ok']}/{accepted_or_checked} = "
            f"{statuses['ok'] / accepted_or_checked:.3f} "
            f"(rejected: contradicts_verdict {details['contradicts_verdict']}, "
            f"attributes_reasoning {details['attributes_reasoning']})")
    if statuses["ok"]:
        cited = sum(1 for r in records if r["llm_status"] == "ok" and r["cited_evidence_ids"])
        log(f"share of accepted LLM explanations citing >= 1 evidence item: "
            f"{cited}/{statuses['ok']} = {cited / statuses['ok']:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-live", action="store_true", help="confirm live LLM calls are intended")
    ap.add_argument("--limit", "--n", dest="limit", type=int, default=MAX_POSTINGS,
                    help=f"number of validation postings, stratified fraud/legit (1-{MAX_POSTINGS}, default {MAX_POSTINGS})")
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS,
                    help=f"seconds between LLM calls (default {DEFAULT_DELAY_SECONDS:g})")
    ap.add_argument("--out", type=Path, default=None, help="results jsonl (default reports/grounding_runs/<model>.jsonl)")
    args = ap.parse_args()
    load_project_env()  # GROQ_API_KEY / LLM_MODEL from .env, if not already set; values are never printed

    if not 1 <= args.limit <= MAX_POSTINGS:
        print(f"--limit must be between 1 and {MAX_POSTINGS}.", file=sys.stderr)
        return 2
    if args.delay < 0:
        print("--delay must be >= 0.", file=sys.stderr)
        return 2
    if not args.run_live:
        print(f"Refusing to run: this makes live LLM calls for {args.limit} postings. "
              "Pass --run-live when intended.", file=sys.stderr)
        return 2

    from backend.routes import job_routes  # loads .env for the key
    from backend.services import dataset as ds
    from backend.services import llm_explainer
    from backend.services.evidence import build_evidence, evidence_ids
    from backend.services.job_analysis import build_job_report

    client = llm_explainer.client_from_env()
    if client is None:
        print("No LLM configured (GROQ_API_KEY / LLM_MODEL missing or ENABLE_LLM_ANALYSIS=false).", file=sys.stderr)
        return 3

    model = os.getenv("LLM_MODEL", "model")
    out_path = args.out or RUNS_DIR / f"grounding_val_{re.sub(r'[^A-Za-z0-9._-]+', '_', model)}.jsonl"
    limited = RateLimitedClient(client)
    breaker = NoBreaker()

    print("=" * 70)
    print("VALIDATION ONLY - grounding evaluation of the LLM explanation (live calls)")
    print(f"model {model} | delay {args.delay:g}s | results {out_path}")
    print("=" * 70)

    val = pd.read_csv(ROOT / "data" / "processed_v2" / "val.csv")
    sample = stratified_sample(val, args.limit)
    classifier = job_routes.get_classifier()
    resources = job_routes.get_evidence_resources()
    guidance_fn = job_routes.get_guidance_fn()

    def analyse(row: dict) -> dict:
        fields = {f: (None if pd.isna(row[f]) else row[f]) for f in ds.TEXT_FIELDS}
        text = ds.build_text(fields)
        profile = None if fields["company_profile"] is None else (ds.clean_field(fields["company_profile"]) or "")
        report = build_job_report(classifier.predict(text), text, profile)
        evidence = build_evidence(text, fields, report["warning_signs"],
                                  posting_index=resources["posting_index"], graph=resources["graph"],
                                  guidance_fn=guidance_fn)
        facts = {k: report[k] for k in ("verdict", "verdict_reason", "display_risk_score")}
        result = llm_explainer.explain(facts, evidence, text, client=limited, breaker=breaker)
        ids = evidence_ids(evidence)
        assert set(result["cited_evidence_ids"]) <= set(ids)
        return {
            "job_id": int(row["job_id"]), "fraudulent": int(row["fraudulent"]), "model": model,
            "verdict": report["verdict"], "llm_status": result["llm_status"], "llm_detail": result["llm_detail"],
            "rate_limit_kind": result["rate_limit_kind"], "explanation_source": result["explanation_source"],
            "cited_evidence_ids": result["cited_evidence_ids"], "n_evidence_ids": len(ids),
            "explanation": result["explanation"],
        }

    result = run_rows(sample.to_dict("records"), analyse, out_path, delay=args.delay,
                      daily_quota_reached=lambda: limited.daily_quota_reached)
    summarize(result)
    print("VALIDATION ONLY.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
