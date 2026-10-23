"""Independently verify a set of split files is FRESH and leakage-free. Never trusts the pipeline's own say-so.

Usage (repo root):
    python scripts/verify_splits.py --dir data/processed_candidate [--max-age-minutes 60]

Reads the split CSVs (including test) ONLY to check integrity; never use this to tune or select models.
Exit code 0 = every check passed, 5 = at least one failed.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services import dataset as ds  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--raw", default=str(ROOT / "data/raw/fake_job_postings.csv"))
    ap.add_argument("--max-age-minutes", type=float, default=None, help="fail if the manifest is older than this")
    args = ap.parse_args()

    results: list[tuple[str, bool, str]] = []
    def check(name, ok, detail=""):
        results.append((name, bool(ok), detail))

    d = Path(args.dir)
    mpath = d / "split_manifest.json"
    if not mpath.exists():
        print(f"[FAIL] manifest: {mpath} not found")
        return 5
    m = json.loads(mpath.read_text(encoding="utf-8"))

    raw_path = Path(args.raw)
    check("raw file present", raw_path.exists(), str(raw_path))
    if not raw_path.exists():
        for n, ok, det in results:
            print(f"[{'PASS' if ok else 'FAIL'}] {n}: {det}")
        return 5
    check("raw sha256 matches manifest", ds.file_sha256(raw_path) == m.get("raw_sha256"))
    check("pipeline code unchanged since the run", ds.file_sha256(ds.__file__) == m.get("pipeline_code_sha256"),
          "re-run prepare_data.py if dataset.py changed after the splits were made")
    check("gate threshold is 0.90", m.get("near_duplicate_threshold") == ds.DEFAULT_NEAR_THRESHOLD,
          f"manifest={m.get('near_duplicate_threshold')}")
    if args.max_age_minutes is not None:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(m["created_at_utc"])).total_seconds() / 60
        check("manifest is recent", age <= args.max_age_minutes, f"{age:.1f} min old")
    check("pipeline reported zero cross-split pairs",
          m.get("near_duplicate_resolution", {}).get("cross_split_pairs_after_split") == 0)

    parts, hashes_ok = {}, True
    for name in ds.SPLIT_NAMES:
        p = d / m["splits"][name]["file"]
        hashes_ok &= p.exists() and ds.file_sha256(p) == m["splits"][name]["sha256"]
        if p.exists():
            parts[name] = pd.read_csv(p)
            parts[name]["text"] = parts[name]["text"].fillna("")  # rows with no text round-trip as NaN
    check("split file hashes match manifest", hashes_ok)

    if len(parts) == 3:
        raw = ds.load_raw(raw_path)
        all_ids = pd.concat([p[ds.ID_COL] for p in parts.values()])
        check("every raw job_id appears exactly once", sorted(all_ids) == sorted(raw[ds.ID_COL]),
              f"{len(all_ids)} split rows vs {len(raw)} raw rows")
        n_fraud = int(sum((p[ds.LABEL_COL] == 1).sum() for p in parts.values()))
        check("fraud count matches raw", n_fraud == int((raw[ds.LABEL_COL] == 1).sum()), f"{n_fraud} fraud rows")
        rep = ds.check_leakage(parts, near_threshold=ds.DEFAULT_NEAR_THRESHOLD)  # recomputed from the files
        for k, c in rep["checks"].items():
            check(f"recomputed: {k}", c["passed"])
        legacy = {k: v["legacy_split_fit_rows_at_or_above"]
                  for k, v in rep["checks"]["no_cross_split_near_duplicates"]["details"].items()}
        print(f"[INFO] legacy split-fit diagnostic (non-gating) rows >= 0.90: {legacy}")

    for n, ok, det in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {n}{': ' + det if det else ''}")
    return 0 if all(ok for _, ok, _ in results) else 5


if __name__ == "__main__":
    sys.exit(main())
