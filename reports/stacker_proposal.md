# Optional Evaluation Stacker: Leakage-Safe Specification (not implemented)

Status: specification only. **Default recommendation: do not implement until Phase 4 shows that retrieval carries signal beyond the classifier.** It is an evaluation device. It never replaces the transparent user-facing 0-100 risk score, and it never feeds LLM judgment or similarity as a probability.

## What it would answer

"Does retrieval information improve fraud detection beyond DistilBERT alone?" Simple ablations (DistilBERT alone; kNN label-vote alone; DistilBERT + rules) already answer part of this. A stacker is only worth adding if it gives a measurably better operating point than the best single signal with uncertainty reported.

## Inputs (4 features, all computed out-of-sample)

1. DistilBERT fraud logit (model trained on train only).
2. Fraction of fraud labels among the top-k retrieved **train** records.
3. Maximum similarity to a retrieved fraud-labelled train record.
4. Weighted rule score (rules fixed before looking at validation labels).

Four features for ~131 validation fraud cases is within the usual guideline of at least ~10 events per feature. The LLM output is **not** an input.

## Leakage-safe procedure

1. Train base models and build the retrieval index from **train only**. Assert no val/test `job_id` or `group_id` is in the index (existing leakage test).
2. Compute the four features for **validation** postings only. These are genuinely out-of-sample (train rows would be unsafe: the classifier has seen them and retrieval would return the row itself).
3. Fit a regularised logistic regression on validation with stratified, **group-aware** 5-fold CV; collect out-of-fold predictions. Choose regularisation strength and the operating threshold from these out-of-fold predictions.
4. Compare, on the same out-of-fold validation predictions: stacker vs DistilBERT alone vs TF-IDF+LR, using fraud-class PR-AUC and recall at a fixed precision, with a paired bootstrap CI on the difference.
5. Accept the stacker only if the improvement's CI excludes zero; otherwise report "no measurable benefit" and drop it.
6. Refit on all validation data, freeze, and apply **once** to the test set (and to the paraphrased test set, unchanged).

## Costs and risks

- Validation is small, so the CI will likely be wide. Report it as is.
- Retrieval index must stay train-only at test time; adding validation records to the index after tuning would change feature distributions.
- More moving parts to explain in the report; justified only by a measured gain.
