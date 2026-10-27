import pytest

flask = pytest.importorskip("flask")
from flask import Flask

from backend.routes import job_routes

THRESHOLD = 0.291163


class FakeClassifier:
    def __init__(self, probability=0.9, error=None):
        self.probability = probability
        self.error = error

    def predict(self, text):
        if self.error:
            raise self.error
        return {"model": "DistilBERT", "prediction": int(self.probability >= THRESHOLD), "label": "x",
                "fraud_probability": self.probability, "legitimate_probability": 1 - self.probability,
                "threshold": THRESHOLD}


@pytest.fixture()
def make_client(monkeypatch):
    def _make(classifier=None, load_error=None):
        def fake_get_classifier():
            if load_error:
                raise load_error
            return classifier

        monkeypatch.setattr(job_routes, "get_classifier", fake_get_classifier)
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.register_blueprint(job_routes.job_bp)
        return app.test_client()

    return _make


def post(client, body):
    return client.post("/api/analyze-job", json=body)


def test_flagged_posting_returns_report(make_client):
    response = post(make_client(FakeClassifier(0.95)), {"text": "Pay a registration fee to start."})
    assert response.status_code == 200
    body = response.get_json()
    assert body["classifier_prediction"] == "Potentially fraudulent"
    assert body["risk_score"] >= 30 and "timeline" not in body


def test_unflagged_posting_returns_lower_risk(make_client):
    body = post(make_client(FakeClassifier(0.01)), {"text": "Software engineer, interview required."}).get_json()
    assert body["risk_category"] == "Lower risk"


@pytest.mark.parametrize("payload", [{}, {"text": ""}, {"text": "   "}, {"text": 123}, {"other": "x"}, [], "text"])
def test_bad_input_is_rejected(make_client, payload):
    response = post(make_client(FakeClassifier()), payload)
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_non_json_body_is_rejected(make_client):
    response = make_client(FakeClassifier()).post("/api/analyze-job", data="plain text", content_type="text/plain")
    assert response.status_code == 400


def test_too_long_text_is_rejected(make_client):
    response = post(make_client(FakeClassifier()), {"text": "a" * (job_routes.MAX_TEXT_CHARS + 1)})
    assert response.status_code == 413


def test_missing_model_gives_503(make_client):
    response = post(make_client(load_error=FileNotFoundError("model dir")), {"text": "A job posting."})
    assert response.status_code == 503
    assert "not installed" in response.get_json()["error"]


def test_load_failure_gives_503(make_client):
    response = post(make_client(load_error=ImportError("torch missing")), {"text": "A job posting."})
    assert response.status_code == 503


def test_prediction_failure_gives_500_without_leaking_text(make_client):
    secret = "my private posting text 12345"
    response = post(make_client(FakeClassifier(error=RuntimeError(secret))), {"text": secret})
    assert response.status_code == 500
    assert secret not in response.get_data(as_text=True)


def test_llm_disabled_returns_classifier_report(make_client, monkeypatch):
    monkeypatch.setenv("ENABLE_LLM_ANALYSIS", "false")

    response = post(
        make_client(FakeClassifier(0.95)),
        {"text": "Pay a registration fee to start."},
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["classifier_prediction"] == "Potentially fraudulent"
    assert body["llm_analysis_status"] == "disabled"
    assert any(
        "feature is disabled" in note.lower()
        for note in body["uncertainty_notes"]
    )


def test_llm_success_adds_analysis_and_evidence(make_client, monkeypatch):
    monkeypatch.setenv("ENABLE_LLM_ANALYSIS", "true")

    fake_result = {
        "model": "test-model",
        "analysis": {
            "assessment": "High risk",
            "explanation": "The posting requests an upfront fee.",
            "warning_signs": ["upfront_fee"],
            "recommended_actions": ["Verify the employer independently."],
            "limitations": ["The posting alone cannot prove fraud."],
        },
        "supporting_evidence": [
            {
                "source_id": "test-source",
                "title": "Job scam guidance",
                "url": "https://example.org/job-scams",
                "excerpt": "Be cautious of requests for upfront payments.",
            }
        ],
    }

    def fake_analyze(text):
        assert "registration fee" in text
        return fake_result

    monkeypatch.setattr(
        "backend.services.llm_analysis.analyze",
        fake_analyze,
    )

    response = post(
        make_client(FakeClassifier(0.95)),
        {"text": "Pay a registration fee to start."},
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["llm_analysis_status"] == "success"
    assert body["llm_model"] == "test-model"
    assert body["warning_signs"] == ["upfront_fee"]
    assert len(body["supporting_evidence"]) == 1
    assert body["explanation"] == fake_result["analysis"]["explanation"]
    assert body["classifier_prediction"] == "Potentially fraudulent"
    assert "The posting alone cannot prove fraud." in body["uncertainty_notes"]


def test_llm_failure_preserves_classifier_report(make_client, monkeypatch):
    monkeypatch.setenv("ENABLE_LLM_ANALYSIS", "true")

    def fake_analyze(text):
        raise RuntimeError("Simulated LLM failure")

    monkeypatch.setattr(
        "backend.services.llm_analysis.analyze",
        fake_analyze,
    )

    response = post(
        make_client(FakeClassifier(0.95)),
        {"text": "Pay a registration fee to start."},
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["classifier_prediction"] == "Potentially fraudulent"
    assert body["llm_analysis_status"] == "unavailable"
    assert any(
        "were unavailable" in note.lower()
        for note in body["uncertainty_notes"]
    )
