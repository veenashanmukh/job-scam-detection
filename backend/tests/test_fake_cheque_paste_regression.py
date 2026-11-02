"""Regression: the fake-cheque text reported as "Job analysis failed (500)" must return 200 with a JSON report on every
input path, with the LLM step off, succeeding, returning bad output, or raising. Fake classifier and fake LLM only."""
import json
from pathlib import Path

import numpy as np
import pytest

flask = pytest.importorskip("flask")
from flask import Flask  # noqa: E402

from backend.routes import job_routes  # noqa: E402

TEXT = ("Title: Remote Office Assistant Description: Earn 900 per week from home. No experience or interview required. "
        "We will mail you a cheque for $2,800. Deposit it, keep $800 as your first week's pay, and transfer the remaining "
        "amount to our equipment vendor")
PROFILE = ("Northgate Logistics is a small family-run courier firm based in Leeds. "
           "We have been inspecting and returning parcels for local retailers since 2015.")
PATHS = {
    "paste, empty profile": ({"text": TEXT, "company_profile": ""}, False),
    "paste, 2-sentence profile": ({"text": TEXT, "company_profile": PROFILE}, True),
    "paste, HTML-only profile": ({"text": TEXT, "company_profile": "<p>&nbsp;</p>"}, False),
    "paste, missing profile key": ({"text": TEXT}, False),
    "form": ({"title": "Remote Office Assistant", "company_profile": "", "description": TEXT.split("Description: ", 1)[1]}, False),
}


class FakeClassifier:
    threshold = 0.291163

    def predict(self, text):
        p = 0.99
        return {"model": "fake", "prediction": 1, "label": "fraudulent", "fraud_probability": p, "legitimate_probability": 1 - p,
                "fraud_logodds": float(np.log(p / (1 - p))), "threshold": self.threshold, "device": "cpu"}


class FakeLLM:
    def __init__(self, mode):
        self.mode = mode

    def complete(self, system, user, timeout):
        if self.mode == "raise":
            raise RuntimeError("provider failure")
        if self.mode == "bad":
            return "not json"
        ids = [item["id"] for item in json.loads(user.split("EVIDENCE (cite only these ids):\n")[1].split("\n\nThe job posting")[0])]
        return json.dumps({"explanation": "You are asked to deposit a cheque and send money on " + "".join(f"[{i}]" for i in ids[:2]) + ".",
                           "cited_evidence_ids": ids[:2], "caveats": []})


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(job_routes, "get_classifier", lambda: FakeClassifier())
    app = Flask(__name__)
    app.testing = True          # an unhandled exception fails the test instead of becoming a 500
    app.register_blueprint(job_routes.job_bp)
    return app.test_client()


@pytest.mark.parametrize("llm", ["off", "ok", "bad", "raise"])
@pytest.mark.parametrize("path", list(PATHS))
def test_the_fake_cheque_text_returns_a_report_on_every_path(client, monkeypatch, path, llm):
    if llm != "off":
        monkeypatch.setattr(job_routes, "get_llm_client", lambda: FakeLLM(llm))
    payload, profile_present = PATHS[path]
    response = client.post("/api/analyze-job", json=payload)
    assert response.status_code == 200, response.get_data(as_text=True)[:500]
    body = response.get_json()
    assert body["verdict"] in {"Needs Verification", "High Fraud Risk"}
    assert {"Cheque or overpayment scam", "No interview"} <= {s["label"] for s in body["warning_signs"]}
    assert body["company_profile_present"] is profile_present
    assert body["explanation"] and isinstance(body["evidence"], dict)
    json.dumps(body)            # the whole report serialises


MODEL_DIR = Path(__file__).resolve().parents[2] / "backend" / "models_v2" / "distilbert_job_scam"


@pytest.mark.skipif(not (MODEL_DIR / "model.safetensors").exists(), reason="the DistilBERT model is not installed")
def test_the_fake_cheque_text_with_the_real_model_returns_200_on_every_path(monkeypatch):
    from backend.services.distilbert_service import DistilBERTJobScamService

    service = DistilBERTJobScamService()
    monkeypatch.setattr(job_routes, "get_classifier", lambda: service)
    app = Flask(__name__)
    app.testing = True
    app.register_blueprint(job_routes.job_bp)
    client = app.test_client()
    for name, (payload, _) in PATHS.items():
        response = client.post("/api/analyze-job", json=payload)
        assert response.status_code == 200, name
        assert response.get_json()["verdict"] == "Needs Verification", name
