"""Grounding-evaluation pacing, resume and label-only rate-limit handling. Fake clients; no network, no data."""
import importlib.util
import json
from pathlib import Path

import pytest

from backend.services import llm_explainer as lx

ROOT = Path(__file__).resolve().parents[2]
SECRET = "Rate limit reached for org_secret123 on tokens per day (TPD): Limit 100000, Used 99999"


@pytest.fixture(scope="module")
def ge():
    spec = importlib.util.spec_from_file_location("eval_grounding_val", ROOT / "scripts" / "eval_grounding_val.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScriptedClient:
    """Raises or returns the scripted outcomes in order."""

    def __init__(self, *outcomes):
        self.outcomes, self.calls = list(outcomes), 0

    def complete(self, system, user, timeout):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


# --- rate-limit labels ------------------------------------------------------------

@pytest.mark.parametrize("message, kind", [
    ("Rate limit reached ... on tokens per day (TPD): Limit 500000", "day"),
    ("Rate limit reached ... on requests per day (RPD)", "day"),
    ("daily quota exceeded", "day"),
    ("Rate limit reached ... on tokens per minute (TPM). Please try again in 7.66s.", "minute"),
    ("requests per minute (RPM) exceeded", "minute"),
    ("Please try again in 950ms", "minute"),
    ("Too many requests", "other"),
    ("", "other"),
])
def test_classify_rate_limit(message, kind):
    assert lx.classify_rate_limit(message) == kind


def test_groq_rate_limit_keeps_only_the_label_and_retry_after(monkeypatch):
    groq = pytest.importorskip("groq")
    httpx = pytest.importorskip("httpx")

    response = httpx.Response(429, headers={"retry-after": "7"},
                              request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"))
    error = groq.RateLimitError(SECRET, response=response, body=None)

    class FakeGroq:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": type("C", (), {"create": staticmethod(self.create)})()})()

        @staticmethod
        def create(**kwargs):
            raise error

    monkeypatch.setattr(groq, "Groq", FakeGroq)
    with pytest.raises(lx.LLMQuotaError) as caught:
        lx.GroqClient("dummy-not-a-real-key", "m").complete("s", "u", timeout=1)

    assert caught.value.kind == "day" and caught.value.retry_after == 7.0
    assert "org_secret123" not in str(caught.value) and caught.value.__cause__ is None


# --- per-minute retries and the daily stop -------------------------------------------

def test_minute_limit_waits_retry_after_then_succeeds(ge):
    sleeps = []
    client = ge.RateLimitedClient(ScriptedClient(lx.LLMQuotaError("minute", 7.0), "ok"), sleep=sleeps.append)
    assert client.complete("s", "u", timeout=1) == "ok"
    assert sleeps == [7.0] and not client.daily_quota_reached


def test_minute_limit_without_retry_after_waits_65s_and_gives_up_after_two_retries(ge):
    sleeps = []
    inner = ScriptedClient(*[lx.LLMQuotaError("minute")] * 3)
    client = ge.RateLimitedClient(inner, sleep=sleeps.append)
    with pytest.raises(lx.LLMQuotaError):
        client.complete("s", "u", timeout=1)
    assert sleeps == [65.0, 65.0] and inner.calls == 3


@pytest.mark.parametrize("kind, daily", [("day", True), ("other", False)])
def test_day_and_other_limits_are_not_retried(ge, kind, daily):
    sleeps = []
    inner = ScriptedClient(lx.LLMQuotaError(kind))
    client = ge.RateLimitedClient(inner, sleep=sleeps.append)
    with pytest.raises(lx.LLMQuotaError):
        client.complete("s", "u", timeout=1)
    assert sleeps == [] and inner.calls == 1 and client.daily_quota_reached is daily


def test_eval_explain_uses_no_breaker_and_records_label_only(ge):
    facts = {"verdict": "High Fraud Risk", "verdict_reason": "r", "display_risk_score": 90}
    evidence = {"similar_postings": [], "graph": {"items": []}, "guidance": [], "rule_signs": []}
    client = ge.RateLimitedClient(ScriptedClient(*[lx.LLMQuotaError("other")] * 5), sleep=lambda s: None)
    for _ in range(5):  # a production breaker would have opened after 3
        result = lx.explain(facts, evidence, "posting", client=client, breaker=ge.NoBreaker())
        assert result["llm_detail"] == "quota" and result["rate_limit_kind"] == "other"
    assert client.client.calls == 5


# --- pacing and resume -------------------------------------------------------------

def _rows(n):
    return [{"job_id": i, "fraudulent": i % 2} for i in range(1, n + 1)]


def _record(row, status="ok", detail="ok", kind=None):
    return {"job_id": row["job_id"], "fraudulent": row["fraudulent"], "llm_status": status, "llm_detail": detail,
            "rate_limit_kind": kind, "cited_evidence_ids": ["R1"] if status == "ok" else []}


def test_calls_are_paced_by_delay_and_each_row_is_saved_as_it_completes(ge, tmp_path):
    out, sleeps, seen = tmp_path / "run.jsonl", [], []

    def analyse(row):
        seen.append(row["job_id"])
        assert len(ge.load_completed(out)) == len(seen) - 1  # earlier rows already on disk
        return _record(row)

    result = ge.run_rows(_rows(4), analyse, out, delay=20, sleep=sleeps.append, log=lambda *a: None)
    assert sleeps == [20, 20, 20]  # between calls, never before the first
    assert seen == [1, 2, 3, 4] and len(result["completed"]) == 4
    assert [json.loads(line)["job_id"] for line in out.read_text().splitlines()] == [1, 2, 3, 4]


def test_resume_skips_finished_job_ids(ge, tmp_path):
    out = tmp_path / "run.jsonl"
    for row in _rows(2):
        ge.append_record(out, _record(row))
    seen = []
    result = ge.run_rows(_rows(4), lambda row: seen.append(row["job_id"]) or _record(row), out,
                         delay=0, sleep=lambda s: None, log=lambda *a: None)
    assert seen == [3, 4] and len(result["completed"]) == 4


def test_daily_quota_stops_cleanly_and_the_next_run_resumes(ge, tmp_path):
    out, logs = tmp_path / "run.jsonl", []
    state = {"daily": False}

    def analyse(row):
        if row["job_id"] == 2:
            state["daily"] = True
            return _record(row, "quota", "quota", "day")
        return _record(row)

    result = ge.run_rows(_rows(3), analyse, out, delay=0, sleep=lambda s: None,
                         daily_quota_reached=lambda: state["daily"], log=logs.append)
    assert ge.DAILY_MESSAGE in logs and result["stopped_daily"]
    assert [r["job_id"] for r in result["completed"]] == [1]

    summary = []
    ge.summarize(result, log=summary.append)
    assert "completed 1 of 3" in summary[0]

    state["daily"], seen = False, []
    rerun = ge.run_rows(_rows(3), lambda row: seen.append(row["job_id"]) or _record(row), out,
                        delay=0, sleep=lambda s: None, log=lambda *a: None)
    assert seen == [2, 3] and len(rerun["completed"]) == 3


def test_rate_limited_and_transient_rows_are_not_saved(ge, tmp_path):
    out = tmp_path / "run.jsonl"
    outcomes = {1: ("ok", "ok", None), 2: ("quota", "quota", "minute"), 3: ("error", "call_failed", None),
                4: ("error", "unknown_evidence_id", None), 5: ("inconsistent", "attributes_reasoning", None)}
    result = ge.run_rows(_rows(5), lambda row: _record(row, *outcomes[row["job_id"]]), out,
                         delay=0, sleep=lambda s: None, log=lambda *a: None)
    assert sorted(r["job_id"] for r in result["completed"]) == [1, 4, 5]
    assert result["not_saved"] == {"minute": 1, "call_failed": 1}


def test_metrics_cover_completed_rows_only(ge):
    rows = [_record({"job_id": 1, "fraudulent": 1}),
            _record({"job_id": 2, "fraudulent": 0}, "inconsistent", "attributes_reasoning"),
            _record({"job_id": 3, "fraudulent": 1}, "error", "unknown_evidence_id")]
    logs = []
    ge.summarize({"completed": rows, "total": 10, "not_saved": {}}, log=logs.append)
    text = "\n".join(logs)
    assert "completed 3 of 10" in text
    assert "citation validity rate: 2/3" in text
    assert "verdict-consistency rate: 1/2" in text and "attributes_reasoning 1" in text
    assert "citing >= 1 evidence item: 1/1" in text


def test_summary_always_reports_completed_even_when_empty(ge):
    logs = []
    ge.summarize({"completed": [], "total": 60, "not_saved": {"day": 1}}, log=logs.append)
    assert logs[0].strip() == "completed 0 of 60"


def test_saved_records_never_contain_provider_error_text(ge, tmp_path):
    out = tmp_path / "run.jsonl"
    facts = {"verdict": "High Fraud Risk", "verdict_reason": "r", "display_risk_score": 90}
    evidence = {"similar_postings": [], "graph": {"items": []}, "guidance": [],
                "rule_signs": [{"id": "R1", "label": "Upfront payment", "matched_text": "fee"}]}
    good = json.dumps({"explanation": "Fee request [R1].", "cited_evidence_ids": ["R1"], "caveats": []})
    minute_secret = "Rate limit reached for org_secret123 on tokens per minute (TPM): Limit 100000. Try again in 2s."
    assert lx.classify_rate_limit(minute_secret) == "minute"
    client = ge.RateLimitedClient(ScriptedClient(lx.LLMQuotaError(lx.classify_rate_limit(minute_secret)), good),
                                  sleep=lambda s: None)

    def analyse(row):
        result = lx.explain(facts, evidence, "posting", client=client, breaker=ge.NoBreaker())
        return {"job_id": row["job_id"], "fraudulent": 1, **{k: result[k] for k in
                ("llm_status", "llm_detail", "rate_limit_kind", "cited_evidence_ids", "explanation")}}

    ge.run_rows(_rows(1), analyse, out, delay=0, sleep=lambda s: None, log=lambda *a: None)
    saved = out.read_text(encoding="utf-8")
    assert '"llm_status": "ok"' in saved and "org_secret123" not in saved and "Limit 100000" not in saved
