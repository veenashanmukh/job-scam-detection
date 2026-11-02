"""Paste mode and form mode build the classifier input and the "profile present" check the same way.
No rule, threshold, tier logic or model weights change."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

flask = pytest.importorskip("flask")
from flask import Flask  # noqa: E402

from backend.routes import job_routes  # noqa: E402
from backend.services.dataset import build_text  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import eval_profile_length_val as pl  # noqa: E402

PARCEL = ("Title: Parcel Inspector, Work From Home. Description: Receive packages at your home address, check them, "
          "and forward them to our overseas customers using the prepaid labels we email you. 25 per parcel, paid weekly.")
PROFILE = ("Northgate Logistics is a small family-run courier firm based in Leeds. "
           "We have been inspecting and returning parcels for local retailers since 2015.")


class TextSensitiveClassifier:
    """Fake classifier whose probability depends on the text it receives, so a profile that reaches the model shows."""

    threshold = 0.291163

    def __init__(self):
        self.seen = []

    def predict(self, text):
        self.seen.append(text)
        p = 0.97 if "Company profile:" not in text else (0.02 if len(text) % 2 else 0.04)
        return {"model": "fake", "prediction": int(p >= self.threshold), "label": "x", "fraud_probability": p,
                "legitimate_probability": 1 - p, "fraud_logodds": float(np.log(p / (1 - p))), "threshold": self.threshold,
                "device": "cpu"}


@pytest.fixture()
def setup(monkeypatch):
    classifier = TextSensitiveClassifier()
    monkeypatch.setattr(job_routes, "get_classifier", lambda: classifier)
    app = Flask(__name__)
    app.register_blueprint(job_routes.job_bp)
    return app.test_client(), classifier


def _post(client, body):
    return client.post("/api/analyze-job", json=body)


def _json(client, body):
    response = _post(client, body)
    assert response.status_code == 200, response.get_data(as_text=True)
    return response.get_json()


# --- the profile reaches the model in paste mode -------------------------------------------------------------------

def test_for_the_parcel_posting_the_profile_reaches_the_model_and_changes_the_probability(setup):
    client, classifier = setup
    none = _json(client, {"text": PARCEL})
    long = _json(client, {"text": PARCEL, "company_profile": PROFILE})
    short = _json(client, {"text": PARCEL, "company_profile": "x"})
    texts = classifier.seen
    assert texts[0] == build_text({"description": PARCEL})
    assert texts[1] == f"Company profile: {PROFILE}\n" + texts[0]
    assert texts[2] == "Company profile: x\n" + texts[0]
    assert len({*texts}) == 3  # three different model inputs
    probabilities = [r["classifier_details"]["fraud_probability"] for r in (none, long, short)]
    assert probabilities[0] > 0.9 and probabilities[1] < 0.1 and probabilities[2] < 0.1
    assert none["verdict_reason"] == "company profile not provided"


def test_paste_mode_gives_exactly_the_output_of_structured_mode_for_the_same_fields(setup):
    client, classifier = setup
    for profile in (None, PROFILE, "x", "", "<p>&nbsp;</p>", "  "):
        paste_body = {"text": PARCEL} if profile is None else {"text": PARCEL, "company_profile": profile}
        form_body = {"description": PARCEL} if profile is None else {"description": PARCEL, "company_profile": profile}
        classifier.seen.clear()
        pasted = _json(client, paste_body)
        form = _json(client, form_body)
        assert classifier.seen[0] == classifier.seen[1], repr(profile)  # same model input
        assert json.dumps(pasted, sort_keys=True) == json.dumps(form, sort_keys=True), repr(profile)  # same full output


# --- the same cleaned profile decides "profile present" -------------------------------------------------------------

@pytest.mark.parametrize("profile", ["", "   ", "<p></p>", "<p> &nbsp; </p>", "&nbsp;&nbsp;", "<br/>"])
def test_html_only_or_blank_profile_counts_as_empty_in_both_modes_and_adds_no_profile_line(setup, profile):
    client, classifier = setup
    pasted = _json(client, {"text": PARCEL, "company_profile": profile})
    form = _json(client, {"description": PARCEL, "company_profile": profile})
    assert all("Company profile:" not in t for t in classifier.seen)
    assert classifier.seen[0] == build_text({"description": PARCEL})  # same text as no profile at all
    for report in (pasted, form):
        assert report["verdict_reason"] == "classifier flagged the posting and the company profile is empty"
        assert report["verdict"] == "Needs Verification"


def test_missing_profile_is_not_provided_but_an_empty_one_is_empty(setup):
    client, _ = setup
    assert _json(client, {"text": PARCEL})["verdict_reason"] == "company profile not provided"
    assert _json(client, {"text": PARCEL, "company_profile": None})["verdict_reason"] == "company profile not provided"
    assert _json(client, {"text": PARCEL, "company_profile": 42})["verdict_reason"] == "company profile not provided"
    assert "company profile is empty" in _json(client, {"text": PARCEL, "company_profile": ""})["verdict_reason"]


def test_a_real_profile_makes_the_verdict_use_the_cleaned_text(setup, monkeypatch):
    client, classifier = setup
    monkeypatch.setattr(classifier, "predict", lambda text: dict(TextSensitiveClassifier().predict("no profile here"), seen=text))
    flagged = _json(client, {"text": PARCEL, "company_profile": "<p>Acme &amp; Co builds tools.</p>"})
    assert flagged["verdict"] == "High Fraud Risk"
    assert flagged["verdict_reason"] == "classifier flagged the posting even though a company profile is present"


def test_pasted_text_that_cleans_to_nothing_and_bad_input_are_rejected_like_before(setup):
    client, classifier = setup
    for body in ({"text": "<p></p>"}, {"text": "   "}, {"text": ""}, {"text": 5}, {}, {"other": "x"}):
        assert _post(client, body).status_code == 400, body
    assert _post(client, {"text": "a" * (job_routes.MAX_TEXT_CHARS + 1)}).status_code == 413
    assert _post(client, [1, 2]).status_code == 400
    assert classifier.seen == []


def test_evidence_fields_match_between_modes(setup, monkeypatch):
    client, _ = setup
    seen = []
    real = job_routes.build_evidence
    monkeypatch.setattr(job_routes, "build_evidence", lambda text, fields, *a, **k: (seen.append((text, dict(fields))), real(text, fields, *a, **k))[1])
    _json(client, {"text": PARCEL, "company_profile": PROFILE})
    _json(client, {"description": PARCEL, "company_profile": PROFILE})
    assert seen[0] == seen[1]
    assert seen[0][1] == {"title": None, "company_profile": PROFILE, "description": PARCEL, "requirements": None, "benefits": None}


# --- the real model, when it is installed ---------------------------------------------------------------------------

MODEL_DIR = Path(__file__).resolve().parents[2] / "backend" / "models_v2" / "distilbert_job_scam"


@pytest.mark.skipif(not (MODEL_DIR / "model.safetensors").exists(), reason="the DistilBERT model is not installed")
def test_with_the_real_model_the_profile_changes_the_probability_and_paste_equals_form(monkeypatch):
    from backend.services.distilbert_service import DistilBERTJobScamService

    service = DistilBERTJobScamService()
    monkeypatch.setattr(job_routes, "get_classifier", lambda: service)
    app = Flask(__name__)
    app.register_blueprint(job_routes.job_bp)
    client = app.test_client()
    probabilities = {}
    for name, profile in (("none", None), ("long", PROFILE), ("x", "x"), ("html", "<p>&nbsp;</p>")):
        body = {"text": PARCEL} if profile is None else {"text": PARCEL, "company_profile": profile}
        form = {"description": PARCEL} if profile is None else {"description": PARCEL, "company_profile": profile}
        pasted, structured = _json(client, body), _json(client, form)
        assert json.dumps(pasted, sort_keys=True) == json.dumps(structured, sort_keys=True), name
        probabilities[name] = pasted["classifier_details"]["fraud_probability"]
    assert len({probabilities["none"], probabilities["long"], probabilities["x"]}) == 3
    assert probabilities["html"] == probabilities["none"]  # an HTML-only profile is the same model input as none


# --- profile-length measurement script ------------------------------------------------------------------------------

def _val():
    rows = [("", 1), (None, 0), ("short", 1), ("x" * 25, 1), ("<p>" + "y" * 120 + "</p>", 1), ("&nbsp;", 0), ("z" * 19, 0), ("w" * 100, 1)]
    return pd.DataFrame([{"job_id": i, "group_id": i, "fraudulent": f, "company_profile": p} for i, (p, f) in enumerate(rows)])


def test_profile_length_buckets_and_fraud_counts_among_flagged():
    val = _val()
    probabilities = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.05, 0.9])  # the 19-character one is not flagged
    result = pl.compute(val, probabilities)
    assert result["n_flagged"] == 7 and result["n_flagged_fraud"] == 5
    assert result["buckets"] == {
        "0": {"n": 3, "fraud": 1, "legitimate": 2},          # "", None and the entity-only profile
        "1-19": {"n": 1, "fraud": 1, "legitimate": 0},       # "short"
        "20-99": {"n": 1, "fraud": 1, "legitimate": 0},      # 25 characters
        "100+": {"n": 2, "fraud": 2, "legitimate": 0},       # 120 characters inside tags, and exactly 100
    }
    assert sum(b["n"] for b in result["buckets"].values()) == result["n_flagged"]


def test_profile_length_report_is_validation_only_and_changes_no_rule():
    lines = pl.report_lines(pl.compute(_val(), np.full(8, 0.9)))
    assert lines[0] == pl.BANNER == lines[-1] and "VALIDATION ONLY" in lines[0]
    assert any(line.startswith("100+ characters") for line in lines)
    assert [pl.bucket_of(n) for n in (0, 1, 19, 20, 99, 100, 5000)] == ["0", "1-19", "1-19", "20-99", "20-99", "100+", "100+"]
    with pytest.raises(ValueError):
        pl.compute(_val(), np.zeros(3))
    source = (Path(__file__).resolve().parents[2] / "scripts" / "eval_profile_length_val.py").read_text(encoding="utf-8")
    assert "job_verdict" not in source and "test.csv" not in source.replace("Uses val.csv", "")


# --- the backend says whether the cleaned profile has any text -----------------------------------------------------------

@pytest.mark.parametrize("profile, present", [
    (None, False), ("", False), ("   ", False), ("<p></p>", False), ("<p> &nbsp; </p>", False), ("&nbsp;&nbsp;", False),
    ("<br/>", False), ("x", True), (PROFILE, True), ("<p>Acme &amp; Co builds tools.</p>", True),
])
def test_company_profile_present_follows_the_cleaned_profile_in_both_modes(setup, profile, present):
    client, _ = setup
    paste = {"text": PARCEL} if profile is None else {"text": PARCEL, "company_profile": profile}
    form = {"description": PARCEL} if profile is None else {"description": PARCEL, "company_profile": profile}
    pasted, structured = _json(client, paste), _json(client, form)
    assert pasted["company_profile_present"] is present and structured["company_profile_present"] is present


def test_the_new_field_is_the_only_addition_and_changes_no_other_field(setup):
    client, _ = setup
    body = _json(client, {"text": PARCEL, "company_profile": PROFILE})
    with_profile = dict(body)
    del with_profile["company_profile_present"]
    from backend.services.job_analysis import build_job_report
    expected = set(build_job_report({"fraud_probability": 0.02, "threshold": 0.291163, "prediction": 0, "fraud_logodds": -3.9,
                                     "model": "x"}, PARCEL, PROFILE))
    assert expected <= set(with_profile)                      # every field build_job_report returns is still there
    assert set(with_profile) - expected <= {"evidence", "explanation_source", "cited_evidence_ids", "llm_caveats", "llm_status"}
    assert body["verdict"] and body["display_risk_score"] is not None


def test_a_non_string_profile_in_paste_mode_is_not_present_and_the_form_rejects_it(setup):
    client, _ = setup
    assert _json(client, {"text": PARCEL, "company_profile": 42})["company_profile_present"] is False   # paste: treated as not provided
    assert _post(client, {"description": PARCEL, "company_profile": 42}).status_code == 400              # form: rejected, as before
