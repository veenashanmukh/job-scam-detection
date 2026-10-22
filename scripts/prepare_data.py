"""Validate, profile, deduplicate and split the EMSCAD file.

Usage (from the repo root):
    python scripts/prepare_data.py --inspect-only          # validation + profile + duplicates, no split written
    python scripts/prepare_data.py                         # full pipeline, writes data/processed/*.csv + manifest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services import dataset, dq_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(ROOT / "data/raw/fake_job_postings.csv"))
    ap.add_argument("--out", default=str(ROOT / "data/processed"))
    ap.add_argument("--report-dir", default=str(ROOT / "reports"))
    ap.add_argument("--near-threshold", type=float, default=dataset.DEFAULT_NEAR_THRESHOLD)
    ap.add_argument("--seed", type=int, default=dataset.SEED)
    ap.add_argument("--inspect-only", action="store_true")
    args = ap.parse_args()

    try:
        result = dataset.run_pipeline(
            args.raw, args.out, near_threshold=args.near_threshold, seed=args.seed, inspect_only=args.inspect_only
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except dataset.SchemaError as e:
        print(f"SCHEMA ERROR: {e}", file=sys.stderr)
        return 3

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "data_quality_report.md").write_text(dq_report.render_markdown(result), encoding="utf-8")
    slim = {k: v for k, v in result.items() if k != "manifest"}
    (report_dir / "data_quality_report.json").write_text(json.dumps(slim, indent=2, default=str), encoding="utf-8")

    for gate, status, detail in dq_report.gates(result):
        print(f"[{status:6}] {gate}: {detail}")
    print(f"\nReport written to {report_dir / 'data_quality_report.md'}")
    if "leakage" in result and not result["leakage"]["all_passed"]:
        print("LEAKAGE CHECK FAILED: do not use these splits.", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
