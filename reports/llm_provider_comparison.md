# LLM Provider Comparison (Gemini vs Groq)

Status: **provisional. One decision input is still missing (Gemini free-tier numbers).** Prepared 2026-10-22 from official documentation only; no API keys were available, so no model was called and **output quality has not been tested**.

## Verified from official docs

| Item | Gemini API | Groq |
|---|---|---|
| Free-tier numbers published in docs | **No.** The rate-limits page lists no free-tier RPM/TPM/RPD; it says limits vary by model and are shown per project in Google AI Studio. | Yes, table on the rate-limits page (see below). The page labels them as base limits for the Free/Developer plans and says to check the account Limits page. |
| Scope of limit | Per project, not per API key. RPD resets at midnight Pacific. | Per organization/account (check account Limits page). |
| Structured JSON output | JSON Schema via `response_format` (`mime_type: application/json` + `schema`). A subset of JSON Schema: `minLength`, `maxLength`, `uniqueItems` etc. are **not enforced** (must be validated in our code). | `json_schema` with `strict: true` (constrained decoding) on `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, `qwen/qwen3.8-27b`, `meta-models/muse-spark-open`; best-effort mode also on `gpt-oss-safeguard-20b`; plain JSON mode otherwise. Structured outputs do not support streaming or tool use. |
| Where to read your real limits | https://aistudio.google.com/rate-limit | Console -> Limits |

Groq text-model limits as published (verify in your account; RPM = requests/min, RPD = requests/day, TPM = tokens/min, TPD = tokens/day):

| Model | RPM | RPD | TPM | TPD |
|---|---|---|---|---|
| openai/gpt-oss-120b | 30 | 1K | 8K | 200K |
| openai/gpt-oss-20b | 30 | 1K | 8K | 200K |
| qwen/qwen3.8-27b | 30 | 1K | 8K | 200K |
| openai/gpt-oss-safeguard-20b | 3 | 1K | 2K | 200K |

## Expected usage (my estimates, not measurements)

Assumed prompt sizes: explanation call about 3,000 input + 500 output tokens (posting + 5 retrieved records + rules); paraphrase call about 600 + 600 tokens. Measure real sizes before relying on these.

| Task | Calls | Approx. tokens |
|---|---|---|
| Paraphrase generation (~130 fraud + ~130 legit, +20% regeneration) | ~310 | ~370K |
| Grounding experiment: 30 postings x 2 conditions (RAG / no retrieval) | 60 | ~180K |
| Development and demo | ~150 | ~450K |
| **Total** | **~520** | **~1.0M** |

Implication for Groq (using the published numbers): the 200K TPD ceiling allows roughly 55-65 explanation-sized calls per day, so ~1.0M tokens is about **5 account-days**. The 8K TPM ceiling also limits explanation calls to about 2 per minute. Request counts (RPD 1K) are not the constraint; **tokens are**. Mitigations: fewer retrieved records (k=3), truncate retrieved text, cache every response, and run experiments in batches across days.

For Gemini, the same table cannot be completed until the AI Studio limits are read.

## Decision rule (to finish today, ~5 minutes)

1. Open https://aistudio.google.com/rate-limit with the team's project. Record, for the model you would use: RPM, TPM, RPD.
2. If Gemini's free limits allow >= ~600 calls/day and >= ~10 RPM, **choose Gemini** (limits comfortably cover the ~520 calls and ~1.0M tokens within 2 days).
3. Otherwise choose **Groq `openai/gpt-oss-120b`** (strict JSON schema available) and plan paraphrase generation and the grounding experiment across several days, with k=3 retrieval.
4. Before committing: send the same 5 real validation postings through the candidate and manually compare JSON validity and faithfulness. Record the outcome here.

I am not recommending a provider yet, because the Gemini side of the comparison is unverified and quality is untested.

## Provider-agnostic integration (to be built in Phase 4)

A single interface, `generate_json(system, user, schema) -> dict`, with one adapter per provider, selected by `LLM_PROVIDER`. Common wrapper responsibilities: response cache keyed by (provider, model, prompt hash); rate limiter driven by the account's actual RPM/TPM; retry with backoff on 429; JSON-schema validation in our code regardless of provider; required evidence-ID checks; deterministic fallback (classifier + rules) when the API fails.

## Sources

- Gemini rate limits: https://ai.google.dev/gemini-api/docs/rate-limits
- Gemini structured output: https://ai.google.dev/gemini-api/docs/structured-output
- Groq rate limits: https://console.groq.com/docs/rate-limits
- Groq structured outputs: https://console.groq.com/docs/structured-outputs
