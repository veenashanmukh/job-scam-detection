"""Prove the app's structured input path matches the training input, on VALIDATION only.

1. Every val.csv row's raw fields are posted to /api/analyze-job (real model, LLM disabled).
   The exact classifier input is compared with val.csv's `text`, and the score with the saved
   val_probabilities.npy.
2. Measurement only: the same rows scored as plain text (cleaned fields joined by newlines,
   without the "Title:" / "Company profile:" ... labels).

Do not point this at the test split.

Usage: python scripts/eval_input_equivalence_val.py [--out results.npz] [--limit N]
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["ENABLE_LLM_ANALYSIS"] = "false"

from flask import Flask  # noqa: E402

from backend.routes import job_routes  # noqa: E402
from backend.services.dataset import TEXT_FIELDS, clean_field  # noqa: E402
from backend.services.distilbert_service import DEFAULT_THRESHOLD as T  # noqa: E402

MD = ROOT / "backend/models_v2/distilbert_job_scam"


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def plain_text(row) -> str:
    parts = [clean_field(row.get(field)) for field in TEXT_FIELDS]
    return "\n".join(part for part in parts if part)


def summarize(name, y, p):
    flag = p >= T
    tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
    print(f"{name:<26} PR-AUC {average_precision_score(y, p):.4f}   recall {tp / max(y.sum(), 1):.4f} "
          f"({tp}/{int(y.sum())})   FP {fp}   precision {tp / max(tp + fp, 1):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="optional .npz to save per-row scores")
    ap.add_argument("--limit", type=int, help="score only the first N rows (smoke test)")
    args = ap.parse_args()

    p_saved = np.load(MD / "val_probabilities.npy")
    y = np.load(MD / "val_labels.npy")
    v = pd.read_csv(ROOT / "data/processed_v2/val.csv")
    assert len(v) == len(p_saved) and (v.fraudulent.values == y).all()
    if args.limit:
        v, p_saved, y = v.iloc[:args.limit], p_saved[:args.limit], y[:args.limit]

    service = job_routes.get_classifier()
    seen = []

    class Recorder:
        def predict(self, text):
            seen.append(text)
            return service.predict(text)

    job_routes.get_classifier = lambda: Recorder()
    app = Flask(__name__)
    app.register_blueprint(job_routes.job_bp)
    client = app.test_client()

    n = len(v)
    p_app, p_plain, flag_app = np.empty(n), np.empty(n), np.empty(n, dtype=bool)
    input_matches = 0
    start = time.time()
    for i, (_, row) in enumerate(v.iterrows()):
        # Raw fields as the API would receive them; NaN becomes JSON null (missing).
        body = {field: (None if pd.isna(row[field]) else row[field]) for field in TEXT_FIELDS}
        response = client.post("/api/analyze-job", json=body)
        assert response.status_code == 200, response.status_code
        report = response.get_json()
        p_app[i] = sigmoid(report["classifier_details"]["fraud_logodds"])
        flag_app[i] = report["classifier_prediction"] == "Potentially fraudulent"
        input_matches += seen[-1] == row["text"]

        p_plain[i] = sigmoid(service.predict(plain_text(row))["fraud_logodds"])

        if (i + 1) % 200 == 0 or i + 1 == n:
            print(f"  scored {i + 1}/{n}  ({time.time() - start:.0f}s)", flush=True)

    diff = np.abs(p_app - p_saved)
    print(f"\n== 1. Structured app path vs saved validation probabilities (n={n}, threshold={T})")
    print(f"classifier input identical to val.csv text: {input_matches}/{n}")
    print(f"max |p_app - p_saved| = {diff.max():.3e}   mean = {diff.mean():.3e}")
    print(f"flag disagreements (app flag vs saved p >= threshold): {int((flag_app != (p_saved >= T)).sum())}")
    print(f"flag disagreements (app p >= threshold vs saved): {int(((p_app >= T) != (p_saved >= T)).sum())}")

    print("\n== 2. Plain text without field labels (measurement only)")
    summarize("structured (app path)", y, p_app)
    summarize("plain text, no labels", y, p_plain)
    print(f"flag disagreements structured vs plain: {int(((p_app >= T) != (p_plain >= T)).sum())}")
    print(f"max |p_plain - p_app| = {np.abs(p_plain - p_app).max():.3f}")

    if args.out:
        np.savez(args.out, p_app=p_app, p_plain=p_plain, p_saved=p_saved, y=y)
        print(f"saved per-row scores to {args.out}")


if __name__ == "__main__":
    main()
