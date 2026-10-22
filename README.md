# RAG-Augmented LLM Framework for Explainable and Robust Job Scam Detection

Research project (Team 7). Primary research question: does a hybrid pipeline (DistilBERT + retrieval + warning-sign rules + evidence-grounded LLM explanation) improve robustness to meaning-preserving paraphrases, and give more evidence-supported explanations, than the TF-IDF+LR and DistilBERT baselines? Outcomes are reported as measured, whatever they are.

**Current status: Phase 1 code complete and tested on a synthetic fixture; NOT yet run on the real dataset.** See "Phase 1 status" below.

## Rules everyone follows

- Seed **42** everywhere. One shared split (`data/processed/`); nobody makes their own.
- The **test split is locked**: `ds.load_split("test", ..., final_evaluation=True)` is the only way to load it, used once in the final evaluation. Never for training, tuning, thresholds, retrieval indexes or model selection.
- Results go in `data/evaluation/results.csv` (`person,experiment,model,metric,value,date`).
- No secrets, private chats, datasets or checkpoints in Git.

## Setup (Windows PowerShell)

```powershell
cd job-scam-detection
py -3.11 -m venv .venv          # any Python >= 3.10; tests were run on 3.13
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

(macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`.)

## Dataset (manual step)

1. Obtain EMSCAD `fake_job_postings.csv` (Kaggle mirror `shivamb/real-or-fake-fake-jobposting-prediction`; original source: University of the Aegean, Vidros et al., *Future Internet* 9(1):6, 2017).
2. **Read the licence/terms yourself** and record the outcome in `reports/data_quality_report.md`. The scripts cannot check it.
3. Save the file as `data/raw/fake_job_postings.csv`.

## Run Phase 1

```powershell
python -m pytest backend/tests -q                 # unit tests (synthetic fixture, no dataset needed)
python scripts/prepare_data.py --inspect-only     # validate + profile + duplicate analysis, writes reports/data_quality_report.*
python scripts/prepare_data.py                    # also writes data/processed/{train,val,test}.csv + split_manifest.json
```

Review `reports/data_quality_report.md` (gates + manual checklist) **before** any model training. Exit codes: 0 ok, 2 file missing, 3 schema error, 4 leakage check failed.

The split is deterministic, but CSVs are not committed (licence/size). One person generates them and shares the files; everyone else verifies `sha256` against the committed `data/processed/split_manifest.json` (`load_split` does this automatically).

## Phase 1 status

| Item | State |
|---|---|
| Repository skeleton, `.gitignore`, `.env.example`, `results.csv` header | done |
| `backend/services/dataset.py` (validation, profiling, dedupe groups, split, leakage checks, manifest, test lock) | implemented, 37 tests passing on **synthetic fixture** |
| `scripts/prepare_data.py` + `reports/data_quality_report` generator | implemented, tested via CLI on fixture |
| `notebooks/01_dataset_inspection.ipynb` | written, schema-valid, **not executed** |
| Real-data validation (licence, schema, label counts, duplicates, redaction) | **not done: dataset was unreachable from the authoring environment** |
| `reports/llm_provider_comparison.md` | provisional; Gemini free-tier limits need an AI Studio check |
| `reports/stacker_proposal.md` | specification only |

## Troubleshooting / fallbacks

- **Dataset missing:** the CLI exits with code 2 and a message; tests still run on the fixture.
- **Slow dedupe:** about 1 min and <1 GB RAM for ~18k documents (measured on synthetic text); raise `--near-threshold` or reduce `chunk` in `dataset.py` if memory is tight.
- **GPU unavailable / model download failures / LLM API down:** relevant from Phase 2 onward; the plan is Colab for training, CPU for TF-IDF, and deterministic classifier+rules output when the LLM is unavailable.
