"""Report how grouping, the resolver and the leakage gate behave on the real file. Writes nothing.

Usage: python scripts/diagnose_near_duplicates.py [--grouping-margin 0.0]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services import dataset as ds  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data/raw/fake_job_postings.csv"))
    ap.add_argument("--grouping-margin", type=float, default=0.0)
    args = ap.parse_args()

    df = ds.add_model_text(ds.load_raw(args.raw)).reset_index(drop=True)
    thr = ds.DEFAULT_NEAR_THRESHOLD
    df[ds.GROUP_COL], st = ds.assign_duplicate_groups(df, near_threshold=thr - args.grouping_margin)
    print(f"initial groups={st['n_groups']} largest={st['largest_group_size']} ({st['largest_group_pct_of_rows']:.2f}%)")

    out, res = ds.resolve_cross_split_near_duplicates(df, near_threshold=thr, grouping_margin=args.grouping_margin)
    print({k: v for k, v in res.items() if k != "merging_pairs_sample"})
    for a, b, s in res["merging_pairs_sample"][:10]:
        print(f"  merged by resolver: {a} <-> {b} cos={s:.6f}")

    rep = ds.check_leakage({n: out[out[ds.SPLIT_COL] == n] for n in ds.SPLIT_NAMES}, near_threshold=thr)
    print("gate passed:", rep["all_passed"])
    for k, v in rep["checks"]["no_cross_split_near_duplicates"]["details"].items():
        print(f"  {k}: max={v['max_similarity']:.6f} rows>=thr={v['n_rows_at_or_above_threshold']} "
              f"legacy(split-fit, non-gating)={v['legacy_split_fit_rows_at_or_above']}")


if __name__ == "__main__":
    main()
