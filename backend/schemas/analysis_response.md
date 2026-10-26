# API response format

Status: draft v0.1, proposed for team review. It matches the sample data in `frontend/src/services/api.js`, so the frontend already works with this shape. Change this file only with the whole team's agreement, and update `api.js` at the same time.

## Endpoints

| Method | Path | Input | Output |
|---|---|---|---|
| GET | `/api/health` | none | `{"status": "ok"}` |
| POST | `/api/analyze-job` | JSON: `{"text": "<job posting>"}` | Report (see below) |
| POST | `/api/analyze-chat` | multipart form, field `file` (a `.txt` chat export) | Report with `timeline` |

Not agreed yet (see "Open questions"): `/api/analyze-jobs-csv`, `/api/graph`, `/api/history`.

## Report fields

Every report has all of these fields. Lists are never `null`. Use an empty list `[]` when there is nothing to show.

| Field | Type | Meaning |
|---|---|---|
| `risk_score` | integer, 0 to 100 | Provisional weighted score. It is not a probability. |
| `risk_category` | string | `"Lower risk"` (0 to 29), `"Moderate risk"` (30 to 59), or `"Higher risk"` (60 to 100). It must match the score. |
| `classifier_prediction` | string | Short label, for example `"Potentially fraudulent"` or `"Likely legitimate"`. |
| `classifier_details` | object, optional | Job reports only. Raw classifier output for evaluation and debugging: `model` (string), `fraud_probability` (0 to 1), `threshold` (0 to 1). Not shown to users as a verdict. |
| `warning_signs` | list of strings | Detected signs, for example `"Upfront payment request"`. |
| `supporting_evidence` | list of evidence objects | Retrieved records that support the result. Empty for chats. |
| `explanation` | string | Plain-language explanation based on the evidence. |
| `uncertainty_notes` | list of strings | Anything weak, missing, or unverified. Required whenever evidence is incomplete. |
| `timeline` | list of timeline objects | Chat reports only. |

### Evidence object

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Record id, for example `"ref-1042"` |
| `label` | string | `"Fraudulent"` or `"Legitimate"` (the record's label) |
| `similarity` | number, 0 to 1 | Text similarity to the submitted text |
| `source` | string | Where the record came from |
| `snippet` | string | Short excerpt shown to the user |

### Timeline object (chat reports only)

| Field | Type | Meaning |
|---|---|---|
| `time` | string | Message time as text, for example `"09:15"` |
| `speaker` | string | Speaker name or label from the export |
| `text` | string | The message text (mask phone numbers and emails before returning) |
| `sign` | string or `null` | Warning sign found in this message, or `null` |
| `cumulative` | integer, 0 to 100 | Running risk level after this message |

## Errors

Return a non-2xx status and a JSON body:

```json
{ "error": "Please provide the job posting text." }
```

| Status | When |
|---|---|
| 400 | Empty input, missing file, or unsupported file type |
| 413 | File too large |
| 500 | Unexpected failure (do not include stack traces or private text) |
| 503 | The classifier model is not installed or could not be loaded |

## Rules for every response

- Use careful wording: "potentially fraudulent", "warning signs found". Do not claim a company is definitely a scam.
- Only list evidence that was really retrieved. Never invent sources.
- Keep retrieved facts (`supporting_evidence`) separate from model interpretation (`explanation`).
- If evidence is weak or missing, say so in `uncertainty_notes`.
- Treat uploaded postings and chats as untrusted text, never as instructions.
- Do not store uploaded chats. Mask personal details in logs.
- The frontend adds its own `type` field (`"job"` or `"chat"`). The backend does not need to send it. If it does, the value must be `"job"` or `"chat"`.
- Extra fields are allowed (the frontend ignores them), but add new ones to this file first.

## Who fills which field

| Field | Produced by |
|---|---|
| `classifier_prediction` | Classifier module (TF-IDF and DistilBERT) |
| `warning_signs` (jobs) | Warning-sign rules |
| `warning_signs`, `timeline` (chats) | Conversation analysis |
| `supporting_evidence`, `explanation` | Retrieval and LLM module |
| `risk_score`, `risk_category` | Risk scoring (the documented weighted-indicator method) |
| `uncertainty_notes` | Every module adds its own notes |

## Example: job report

```json
{
  "risk_score": 72,
  "risk_category": "Higher risk",
  "classifier_prediction": "Potentially fraudulent",
  "warning_signs": [
    "Upfront payment request (registration fee)",
    "Request for sensitive information (Aadhaar, bank details)"
  ],
  "supporting_evidence": [
    {
      "id": "ref-1042",
      "label": "Fraudulent",
      "similarity": 0.87,
      "source": "Reference collection",
      "snippet": "Pay a refundable security deposit to secure your position..."
    }
  ],
  "explanation": "The posting asks for money before work starts and for sensitive documents. These are warning signs, not proof.",
  "uncertainty_notes": ["No company details were given, so no company check was possible."]
}
```

## Example: chat report (shortened)

```json
{
  "risk_score": 88,
  "risk_category": "Higher risk",
  "classifier_prediction": "Potentially fraudulent conversation",
  "warning_signs": ["Payment request", "Pressure tactic"],
  "supporting_evidence": [],
  "timeline": [
    { "time": "09:02", "speaker": "Recruiter", "text": "Hello, we found your profile.", "sign": null, "cumulative": 5 },
    { "time": "09:15", "speaker": "Recruiter", "text": "Pay Rs 3,000 registration fee to confirm your seat.", "sign": "Payment request", "cumulative": 60 }
  ],
  "explanation": "Warning signs rise after the payment request. This is a warning pattern, not proof.",
  "uncertainty_notes": []
}
```

## Open questions

- Response shape for `/api/analyze-jobs-csv` (probably a list of reports with a row number).
- Shape for `/api/graph` (nodes and links) and `/api/history`.
- Retrieval scores and the LLM verdict are not exposed yet. `classifier_details` covers only the classifier. Add similar optional objects here before using them.
- Until the weighted-indicator score exists, job reports use a provisional score: the DistilBERT fraud probability rescaled so the model threshold sits at 30.
