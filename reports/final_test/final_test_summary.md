# Final test evaluation

**FINAL TEST EVALUATION. This is the second look at the test split; an earlier evaluation exists (reports/distilbert_v1/test_results.json).**

Generated 2026-11-01T10:56:34+00:00 from commit 29d90bbef7ceaf5fa083158845f7529eb9406f39 (script: tracked, unmodified). Test rows 2695, fraud 144. 95% intervals: group bootstrap by group_id, 2000 resamples, seed 42.

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | TP / FP / FN / TN |
|---|---|---|---|---|---|---|
| distilbert | 0.855 (0.721-0.966) | 0.985 (0.968-0.997) | 0.764 (0.564-0.916) | 0.764 (0.537-0.939) | 0.764 (0.593-0.899) | 110 / 34 / 34 / 2517 |
| baseline | 0.802 (0.657-0.942) | 0.985 (0.970-0.995) | 0.716 (0.530-0.859) | 0.771 (0.543-0.942) | 0.742 (0.577-0.875) | 111 / 44 / 33 / 2507 |
| combiner_exploratory (exploratory, frozen on validation) | 0.765 (0.552-0.972) | 0.989 (0.974-0.998) | 0.789 (0.580-0.938) | 0.778 (0.547-0.950) | 0.783 (0.611-0.920) | 112 / 30 / 32 / 2521 |

Paired PR-AUC difference, DistilBERT minus baseline: +0.052 (95% -0.022 to +0.139); above zero in 89.8% of resamples.

## Verdict tiers

| Tier | Postings | Fraud | Fraud rate (Clopper-Pearson 95%) |
|---|---|---|---|
| Likely Legitimate | 2551 | 34 | 0.013 (0.009-0.019) |
| Needs Verification | 102 | 85 | 0.833 (0.747-0.900) |
| High Fraud Risk | 42 | 25 | 0.595 (0.433-0.744) |

## Calibration

Temperature T = 2.004 (fitted on validation only).

| Metric | Before | After |
|---|---|---|
| ECE (15 bins) | 0.024 (0.008-0.045) | 0.010 (0.005-0.034) |
| Log-loss | 0.150 (0.045-0.309) | 0.090 (0.037-0.169) |

The combiner row is exploratory, frozen on validation. Nothing was tuned on test. This is a second look at the test split.
