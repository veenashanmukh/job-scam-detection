"""Print duplicate groups whose members have conflicting labels (READ-ONLY: changes no data, writes one report CSV).

Usage (from the repo root):
    python scripts/inspect_conflicts.py
Takes about a minute on the full dataset (it recomputes the duplicate groups).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from backend.services import dataset as ds  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # CMD-safe printing
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=str(ROOT / "data/raw/fake_job_postings.csv"))
    ap.add_argument("--out", default=str(ROOT / "reports/conflict_groups.csv"))
    ap.add_argument("--near-threshold", type=float, default=ds.DEFAULT_NEAR_THRESHOLD)
    args = ap.parse_args()

    raw = ds.load_raw(args.raw)
    ds.validate_schema(raw)
    df = ds.add_model_text(raw)
    df[ds.GROUP_COL], _ = ds.assign_duplicate_groups(df, near_threshold=args.near_threshold)

    n_labels = df.groupby(ds.GROUP_COL)[ds.LABEL_COL].nunique()
    conflict_ids = n_labels[n_labels > 1].index.tolist()
    print(f"Groups with conflicting labels: {len(conflict_ids)}")
    if not conflict_ids:
        return 0

    df["_norm"] = df["text"].map(ds.normalize_for_dedupe)
    rows = []
    for gid in conflict_ids:
        g = df[df[ds.GROUP_COL] == gid].sort_values(ds.ID_COL)
        identical = g["_norm"].nunique() == 1
        n_fraud = int((g[ds.LABEL_COL] == 1).sum())
        print("\n" + "=" * 78)
        print(f"GROUP {gid}: {len(g)} postings | fraud={n_fraud} legit={len(g) - n_fraud} | "
              f"normalised text identical across all members: {identical}")
        for _, r in g.iterrows():
            title = clean = ds.clean_field(r.get("title")) or ""
            desc = (ds.clean_field(r.get("description")) or "")[:140]
            company = (ds.clean_field(r.get("company_profile")) or "")[:80]
            print(f"  job_id={r[ds.ID_COL]:<6} (CSV line {int(r[ds.ID_COL]) + 1}) label={int(r[ds.LABEL_COL])} "
                  f"| {title[:60]} | {r.get('location')}")
            print(f"      company: {company}")
            print(f"      desc:    {desc}")
            rows.append({
                "group_id": gid, "group_size": len(g), "text_identical_in_group": identical,
                ds.ID_COL: r[ds.ID_COL], "csv_line_number": int(r[ds.ID_COL]) + 1, ds.LABEL_COL: int(r[ds.LABEL_COL]),
                "title": title, "location": r.get("location"), "company_profile": ds.clean_field(r.get("company_profile")),
                "description": ds.clean_field(r.get("description")), "requirements": ds.clean_field(r.get("requirements")),
                "benefits": ds.clean_field(r.get("benefits")),
            })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")  # utf-8-sig so Excel opens it correctly
    print(f"\nFull text written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())