
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services import dataset as ds

PAIRS = [
    (16193, 11441),
    (17136, 17466),
    (3073, 5816),
    (10389, 10770),
]

raw = ds.load_raw(ROOT / "data/raw/fake_job_postings.csv")
df = ds.add_model_text(raw).set_index(ds.ID_COL)

fields = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]

for a_id, b_id in PAIRS:
    a = df.loc[a_id]
    b = df.loc[b_id]

    print("\n" + "=" * 80)
    print(f"JOB {a_id}: {a.get('title', '')} | {a.get('location', '')}")
    print(f"JOB {b_id}: {b.get('title', '')} | {b.get('location', '')}")

    for field in fields:
        left = str(a.get(field, "") or "").strip()
        right = str(b.get(field, "") or "").strip()

        if not left and not right:
            continue

        print(f"\n--- {field} ---")
        print(f"JOB {a_id}: {left[:700]}")
        print(f"JOB {b_id}: {right[:700]}")
