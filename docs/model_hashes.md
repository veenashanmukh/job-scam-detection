# Model and cache file hashes

These files are not committed (model weights and derived caches are large or regenerable; see .gitignore).
Their SHA-256 hashes are recorded here so a copy can be checked against the version used for the results.
Recorded 2026-11-01 from the working copy.

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `backend/models_v2/baseline_tfidf_lr.joblib` | 15,527,035 | `4fcc1c217f3aea82bfdd6ea052fec11d33e4a7a758fffd01560fd4e9267bba24` |
| `backend/models_v2/distilbert_job_scam/config.json` | 593 | `c16465b55ad55217d1f0d39552ef5fda1bac82fb1540a32907a2f63d9ecac629` |
| `backend/models_v2/distilbert_job_scam/model.safetensors` | 267,832,560 | `82385cb7644ced31cbe1454326b9437c164ecf25b9f782834284a253b9de298f` |
| `backend/models_v2/distilbert_job_scam/tokenizer.json` | 711,494 | `435667fab0c06c165b1283ecb422497c37124f2d6a35b2ac73dc876332fc9518` |
| `backend/models_v2/distilbert_job_scam/tokenizer_config.json` | 351 | `e1c2a61a99bda00f6c55303a210b30e2f92dcf8b555e215812e2eb583e177ffd` |
| `backend/models_v2/distilbert_job_scam/training_history.csv` | 327 | `7540b001bdbb261eed6424c7ab355c7c3361ea672e0222cb26249b4d70ba10aa` |
| `backend/models_v2/distilbert_job_scam/val_labels.npy` | 21,576 | `c72fa441c828c9f55ed495f2bca687be3e6feb635432e18c92f2a737bb80697b` |
| `backend/models_v2/distilbert_job_scam/val_probabilities.npy` | 21,576 | `66643cae680ea6c8b80c0efdf78f0cae8d7d86e6e3513e745bb5fb026396d38f` |
| `backend/models_v2/entity_graph.pkl` | 1,585,642 | `79b04160e209edef7301d7956d5e3563129296761546e36f5f3a614b68b6550b` |
| `backend/models_v2/posting_index.pkl` | 95,678,737 | `a38bb7bb56eea2e31e4cb7234b2e333c7f013284a13d807282eb0540dba129e6` |

## Data splits (from data/processed_v2/split_manifest.json)

The split files are not committed either. Hashes as recorded in the manifest:

| Split | Rows | Fraud | SHA-256 |
|---|---:|---:|---|
| train | 12,504 | 594 | `772a9abbce5a37e324129335f52de72c29b4c40e9b6e4b15bee2ab1a1fd3f54c` |
| val | 2,681 | 128 | `c0b3ce75ef772d6fb0289776119c1a8188560505491a24fc9c752c7dcecf5cf0` |
| test | 2,695 | 144 | `46e46b3e91197f8e0749028375be2ca7957f251d1a0ee4a10d2565565082492f` |

The test split is locked: `scripts/final_test_eval.py --preflight` checks these hashes without reading test rows.
