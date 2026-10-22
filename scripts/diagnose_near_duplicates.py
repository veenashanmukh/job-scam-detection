
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services import dataset as ds


def main():
    raw = ds.load_raw(ROOT / "data/raw/fake_job_postings.csv")
    df = ds.add_model_text(raw).reset_index(drop=True)

    groups, _ = ds.assign_duplicate_groups(
        df, near_threshold=ds.DEFAULT_NEAR_THRESHOLD
    )
    df[ds.GROUP_COL] = groups

    df[ds.SPLIT_COL] = ds.group_stratified_split(
        df, fractions=ds.DEFAULT_FRACTIONS, seed=ds.SEED
    )

    splits = {
        name: df[df[ds.SPLIT_COL] == name]
        for name in ds.SPLIT_NAMES
    }

    threshold = ds.DEFAULT_NEAR_THRESHOLD

    for later, earlier in (
        ("val", "train"),
        ("test", "train"),
        ("test", "val"),
    ):
        a_df = splits[later]
        b_df = splits[earlier]

        a_all = a_df["text"].map(ds.normalize_for_dedupe).tolist()
        b_all = b_df["text"].map(ds.normalize_for_dedupe).tolist()

        # Keep original row positions so we can print the offending jobs.
        a_positions = [i for i, text in enumerate(a_all) if text]
        b_positions = [i for i, text in enumerate(b_all) if text]
        a_texts = [a_all[i] for i in a_positions]
        b_texts = [b_all[i] for i in b_positions]

        print(f"\n=== {later} vs {earlier} ===")

        if not a_texts or not b_texts:
            print("One split has no non-empty text.")
            continue

        # This is the same similarity calculation used by check_leakage.
        maxima = ds.max_cross_similarity(a_texts, b_texts)

        hit_indices = np.flatnonzero(maxima >= threshold)

        print(f"Rows at/above threshold {threshold}: {len(hit_indices)}")
        print(f"Maximum similarity: {maxima.max():.6f}")

        if len(hit_indices) == 0:
            print("No pairs above threshold.")
            continue

        # Recalculate similarity for the flagged rows, in manageable chunks.
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.feature_extraction.text import TfidfVectorizer

        combined = a_texts + b_texts
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            dtype=np.float64,
        )
        X = vectorizer.fit_transform(combined)
        XA, XB = X[:len(a_texts)], X[len(a_texts):]

        shown = 0
        for start in range(0, len(hit_indices), 20):
            batch = hit_indices[start:start + 20]
            sims = cosine_similarity(XA[batch], XB)

            for local_i, row_sims in enumerate(sims):
                j = int(row_sims.argmax())
                i = int(batch[local_i])
                sim = float(row_sims[j])

                a_row = a_df.iloc[a_positions[i]]
                b_row = b_df.iloc[b_positions[j]]

                print(
                    f"\nsimilarity={sim:.6f}\n"
                    f"  {later}: job_id={a_row[ds.ID_COL]} "
                    f"label={a_row[ds.LABEL_COL]} "
                    f"title={a_row.get('title', '')!r} "
                    f"location={a_row.get('location', '')!r}\n"
                    f"  {earlier}: job_id={b_row[ds.ID_COL]} "
                    f"label={b_row[ds.LABEL_COL]} "
                    f"title={b_row.get('title', '')!r} "
                    f"location={b_row.get('location', '')!r}"
                )

                shown += 1
                if shown >= 10:
                    break

            if shown >= 10:
                break


if __name__ == "__main__":
    main()
