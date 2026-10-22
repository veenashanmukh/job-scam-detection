
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services import dataset as ds

PAIRS = [
    ("val", "train", 16193, 11441, 0.901267),
    ("val", "train", 17136, 17466, 0.904304),
    ("test", "val", 3073, 5816, 0.901006),
    ("test", "val", 10389, 10770, 0.906436),
]

def main():
    raw = ds.load_raw(ROOT / "data/raw/fake_job_postings.csv")
    df = ds.add_model_text(raw).set_index(ds.ID_COL)

    rows = []
    for split_a, split_b, id_a, id_b, similarity in PAIRS:
        a = df.loc[id_a]
        b = df.loc[id_b]

        rows.append({
            "split_a": split_a,
            "job_id_a": id_a,
            "title_a": a.get("title", ""),
            "location_a": a.get("location", ""),
            "label_a": int(a[ds.LABEL_COL]),
            "split_b": split_b,
            "job_id_b": id_b,
            "title_b": b.get("title", ""),
            "location_b": b.get("location", ""),
            "label_b": int(b[ds.LABEL_COL]),
            "similarity": similarity,
            "review_status": "PENDING",
            "review_notes": "",
        })

    out = ROOT / "reports" / "near_duplicate_review.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved review report: {out}")
    print(f"Pairs to review: {len(rows)}")

if __name__ == "__main__":
    main()
