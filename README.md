# Evidence-Grounded Detection of Fraudulent Job Postings and Recruiter Conversations

RV University, School of Computer Science and Engineering. Team 7. Guide: Prof. Lokesh J K.

A job-scam analysis system. A user pastes a job posting, fills in a structured posting, or pastes a recruiter chat (WhatsApp, Telegram or plain text). The system returns a risk score from 1 to 99, a verdict, plain-language reasons tied to numbered evidence, and practical next steps.

> The score is a risk indicator. It is **not** a probability, and it is never exactly 0 or 100. The system is a cautious first check that helps a person decide what to verify. It is not a verdict that clears or condemns an offer.

## What is in the system

| Part | What it does | Where |
|---|---|---|
| Classifier | DistilBERT fine-tuned on EMSCAD, calibrated with temperature scaling (T = 2.0043), decision threshold 0.291163 | `backend/services/distilbert_service.py`, `score_display.py` |
| Baseline | TF-IDF + logistic regression (C = 30) for comparison | `backend/services/baseline.py`, `scripts/train_baseline.py` |
| Rules | Scam-pattern categories (cheque or overpayment, gift card or crypto, parcel forwarding, upfront payment, personal-information requests, no interview, and others) | `backend/services/job_analysis.py`, `conversation_analysis.py` |
| Verdict | Likely Legitimate, Needs Verification, High Fraud Risk, with rule escalation that never lowers a verdict and never changes the score | `backend/services/job_analysis.py` |
| Entity graph | Links a posting's company profile and hashed contact tokens to labelled training postings | `backend/services/entity_graph.py` |
| Explanation | Evidence items (E1..En, G1..Gn) given to a language model through the Groq API, with a template fallback so every analysis has an explanation | `backend/services/evidence.py`, `llm_explainer.py` |
| Chat analysis | Parsers for WhatsApp, Android, Telegram, paragraph and untimestamped chats, plus chat rules | `backend/services/conversation_parser.py`, `conversation_analysis.py` |
| Web app | React single-page app: verdict, reasons, warning signs, next steps | `frontend/` |

## Results at a glance

Final test evaluation, run once on 1 November 2026 from commit `29d90bb`. The test split has 2,695 postings (144 fraudulent) and had already been evaluated once in October, so this is a second look. Intervals are 95% group bootstrap intervals.

| Model | PR-AUC | F1 | TP / FP / FN / TN |
|---|---|---|---|
| DistilBERT | 0.855 (0.721 to 0.966) | 0.764 | 110 / 34 / 34 / 2517 |
| TF-IDF + LR baseline | 0.802 (0.657 to 0.942) | 0.742 | 111 / 44 / 33 / 2507 |
| Similarity combiner (exploratory) | 0.765 (0.552 to 0.972) | 0.783 | 112 / 30 / 32 / 2521 |

- DistilBERT minus baseline in PR-AUC is +0.052 (95% interval -0.022 to +0.139). The data favour DistilBERT but do not establish that it is better.
- Temperature scaling lowers expected calibration error from 0.024 to 0.010 on test.
- **Did not replicate:** the similarity combiner (gain on validation, lower PR-AUC on test), and the verdict-tier ordering. On test, High Fraud Risk had a fraud rate of 59.5% (25 of 42) and Needs Verification 83.3% (85 of 102), so the tier names must not be read as a ranking.
- Full numbers and caveats are in the project report. Supporting experiments (paraphrase robustness, company-profile injection, explanation grounding, chat rules) are run on validation data or small hand-written sets and are labelled as such.

## Known limitations

- A copied or generic company profile can hide a fraudulent posting from the classifier. One defence was tested against a pre-registered bar and rejected.
- Chat analysis is rule-based. Recall was 0.333 (precision 0.80) on a small hand-written set. It has not been validated at scale.
- EMSCAD is an older dataset. Newer scam styles (for example cryptocurrency or task-based schemes) are covered only by a few hand-written cases.
- Explanations were checked automatically on 60 postings and manually on 20, by one reviewer. The grounding run was made before a later correction to per-entity link counts.
- In the live app, only the company profile (and dataset tokens, where present) can be matched in the entity graph. Contact details typed by a user are not matched.

## Repository layout

```
backend/
  app.py              Flask app (python -m backend.app)
  routes/             job_routes.py, chat_routes.py
  services/           classifier, rules, verdict, entity graph, evidence, explainer, chat analysis
  data/               demo_examples.json and small evaluation records
  tests/              Python tests
frontend/             React + Vite app
scripts/              data preparation, training, evaluation and the guarded final test
data/                 synthetic dev sets (synthetic_jobs, synthetic_chats); data/raw and processed splits are not committed
docs/model_hashes.md  SHA-256 hashes of model, cache and split files
reports/              data-quality and earlier evaluation reports
notebooks/            dataset inspection notebook
```

## Files that are not in Git

These must exist locally before the app or the evaluation scripts run. They are ignored by `.gitignore` because of size, licence or secrecy.

| What | Where | Notes |
|---|---|---|
| EMSCAD raw file | `data/raw/fake_job_postings.csv` | Obtain it yourself and read the licence terms (see Dataset). |
| Split files | `data/processed_v2/{train,val,test}.csv` and `split_manifest.json` | Seed 42, group-aware 70/15/15. Verify hashes against `docs/model_hashes.md`. |
| Models and caches | `backend/models_v2/` (`distilbert_job_scam/`, `baseline_tfidf_lr.joblib`, `entity_graph.pkl`, `posting_index.pkl`) | The DistilBERT model was trained on Kaggle; its notebook is not in this repository. Hashes are in `docs/model_hashes.md`. The two `.pkl` caches can be rebuilt with `python scripts/build_evidence_indexes.py`. |
| Secrets | `.env` | Never commit it. See Setup. |

Ask the project owner for the split files and the model folder, then check each file's SHA-256 against `docs/model_hashes.md` before using it.

## Setup (Windows PowerShell)

```powershell
cd job-scam-detection
py -3.11 -m venv .venv              # any Python 3.10 or newer
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd frontend
npm install
cd ..
```

Create a `.env` file in the project root (copy `.env.example`). The explanation layer needs:

```
GROQ_API_KEY=<your key>
LLM_MODEL=<a model available to your Groq account>
```

Without a key, or when the free-tier quota is used up, the app still works and shows the template explanation.

## Run

Use two terminals and keep both open.

```powershell
# Terminal 1: backend (http://127.0.0.1:5000)
.\.venv\Scripts\Activate.ps1
python -m backend.app

# Terminal 2: frontend (usually http://localhost:5173)
cd frontend
npm run dev
```

In development, the frontend proxies `/api` to the backend. If the page says "Could not reach the analysis server", the backend is not running.

Endpoints: `POST /api/analyze-job`, `POST /api/analyze-chat`, `GET /api/demo-examples`.

## Tests

```powershell
python -m pytest backend/tests -q     # about 960 tests at commit 29d90bb
cd frontend
npm test                              # 99 tests at commit 29d90bb
npm run build
```

## Evaluation scripts and the test-set rule

All scripts run from the project root with the virtual environment active.

- `scripts/prepare_data.py`: validate, profile, deduplicate and split EMSCAD.
- `scripts/verify_splits.py`: confirm splits are fresh and leakage-free.
- `scripts/train_baseline.py`: train the TF-IDF + logistic-regression baseline.
- `scripts/eval_*.py`: validation-only experiments (paraphrase, profile injection, grounding, chat rules, verdict tiers, and others). They never read the test split.
- `scripts/final_test_eval.py`: the guarded final evaluation of the test split.

**The test split is locked.** Do not use it for training, tuning, thresholds, retrieval indexes or model selection. The final evaluation has already been run (see Results). To check the setup without reading any test rows, run:

```powershell
python scripts/final_test_eval.py --preflight
```

Do not run `--final-evaluation` again.

## Dataset

EMSCAD (Employment Scam Aegean Dataset), 17,880 postings, 866 labelled fraudulent. Source: University of the Aegean, Vidros, Kolias, Kambourakis and Akoglu, *Future Internet* 9(1):6, 2017. A Kaggle mirror exists (`shivamb/real-or-fake-fake-jobposting-prediction`). Read the licence terms yourself and record the outcome in `reports/data_quality_report.md`.

## Contributor rules

- Seed 42 everywhere. One shared split; nobody makes their own.
- No secrets, private chats, datasets or model weights in Git.
- Results are reported as measured, including negative ones.
- Experiments that were not part of the main protocol are labelled exploratory.
