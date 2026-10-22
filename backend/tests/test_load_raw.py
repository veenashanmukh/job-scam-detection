import pandas as pd
import pytest

from backend.services import dataset as ds


def _write(tmp_path, labels, with_id=False):
    n = len(labels)
    df = pd.DataFrame({
        "title": [f"t{i}" for i in range(n)],
        "company_profile": [None] * n,
        "description": [f"d{i}" for i in range(n)],
        "requirements": [None] * n,
        "benefits": [None] * n,
        "fraudulent": labels,
    })
    if with_id:
        df.insert(0, "job_id", [100 + i for i in range(len(labels))])
    p = tmp_path / "x.csv"
    df.to_csv(p, index=False)
    return p


@pytest.mark.parametrize("labels", [
    ["f", "t", "f", "t"],                   # EMSCAD style
    ["F", " T ", "f", "t"],                 # case and whitespace
    ["false", "true", "False", "TRUE"],     # words
    [False, True, False, True],             # pandas reads this as bool
    [0, 1, 0, 1],                           # ints
    [0.0, 1.0, 0.0, 1.0],                   # floats
])
def test_load_raw_normalises_label_representations(tmp_path, labels):
    out = ds.load_raw(_write(tmp_path, labels))
    assert out[ds.LABEL_COL].tolist() == [0, 1, 0, 1]
    assert str(out[ds.LABEL_COL].dtype) == "int8"


@pytest.mark.parametrize("dtype", ["object", "string", "category"])
def test_normalize_labels_handles_pandas_dtypes(dtype):
    s = pd.Series(["f", "t", "f"], name="fraudulent").astype(dtype)
    assert ds.normalize_labels(s).tolist() == [0, 1, 0]


def test_unknown_label_is_rejected_with_the_offending_value(tmp_path):
    with pytest.raises(ds.SchemaError, match="maybe"):
        ds.load_raw(_write(tmp_path, ["f", "t", "maybe"]))


def test_missing_label_is_rejected(tmp_path):
    with pytest.raises(ds.SchemaError, match="missing"):
        ds.load_raw(_write(tmp_path, ["f", None, "t"]))


def test_job_id_generated_in_row_order_when_absent(tmp_path):
    out = ds.load_raw(_write(tmp_path, ["f", "t", "f"]))
    assert out[ds.ID_COL].tolist() == [1, 2, 3]
    assert out.columns[0] == ds.ID_COL


def test_existing_job_id_is_preserved(tmp_path):
    out = ds.load_raw(_write(tmp_path, ["f", "t"], with_id=True))
    assert out[ds.ID_COL].tolist() == [100, 101]


def test_normalised_output_passes_schema_validation(tmp_path):
    out = ds.load_raw(_write(tmp_path, ["f", "t", "f"]))
    assert ds.validate_schema(out)["n_rows"] == 3