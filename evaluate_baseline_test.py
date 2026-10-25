import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "backend" / "models_v2" / "baseline_tfidf_lr.joblib"
TEST_PATH = ROOT / "data" / "processed_v2" / "test.csv"
OUTPUT_PATH = ROOT / "reports" / "baseline_v2" / "baseline_test_results.json"

# Load the existing model and its frozen validation threshold.
artifact = joblib.load(MODEL_PATH)
pipeline = artifact["pipeline"]
threshold = float(artifact["threshold"])

# Load the locked test split.
test_df = pd.read_csv(TEST_PATH)
X_test = test_df["text"].fillna("").astype(str).tolist()
y_test = test_df["fraudulent"].to_numpy(dtype=int)

if len(np.unique(y_test)) != 2:
    raise ValueError("Test split must contain both classes.")

# Logistic Regression scores are decision_function values, not probabilities.
scores = pipeline.decision_function(X_test)
predictions = (scores >= threshold).astype(int)

results = {
    "model": "tfidf_lr",
    "evaluated_split": "test",
    "n_test": int(len(y_test)),
    "n_legitimate": int((y_test == 0).sum()),
    "n_fraudulent": int((y_test == 1).sum()),
    "frozen_validation_threshold": threshold,
    "accuracy": float(accuracy_score(y_test, predictions)),
    "precision": float(precision_score(y_test, predictions, zero_division=0)),
    "recall": float(recall_score(y_test, predictions, zero_division=0)),
    "f1": float(f1_score(y_test, predictions, zero_division=0)),
    "average_precision_pr_auc": float(average_precision_score(y_test, scores)),
    "roc_auc": float(roc_auc_score(y_test, scores)),
    "confusion_matrix": confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).tolist(),
    "classification_report": classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Legitimate", "Fraudulent"],
        output_dict=True,
        zero_division=0,
    ),
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

print("\nLOGISTIC REGRESSION — TEST RESULTS")
print("=" * 45)
print(f"Test examples: {results['n_test']}")
print(f"Frozen validation threshold: {threshold:.6f}")
print(f"Accuracy:  {results['accuracy']:.4f}")
print(f"Precision: {results['precision']:.4f}")
print(f"Recall:    {results['recall']:.4f}")
print(f"F1-score:  {results['f1']:.4f}")
print(f"PR-AUC:    {results['average_precision_pr_auc']:.4f}")
print(f"ROC-AUC:   {results['roc_auc']:.4f}")
print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
print(np.array(results["confusion_matrix"]))
print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Legitimate", "Fraudulent"],
    zero_division=0,
))
print(f"Results saved to: {OUTPUT_PATH}")
