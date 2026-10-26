"""Smoke test with the REAL model. Skipped automatically when torch or the model files are missing."""
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
from backend.services import distilbert_service

if not distilbert_service.MODEL_DIR.exists():
    pytest.skip("DistilBERT model folder not found", allow_module_level=True)

SCAM = ("Work from home data entry. Earn $5,000 every week, no interview needed. "
        "Pay a registration fee before joining and send the payment immediately to reserve your position.")
NORMAL = ("Software Engineer, Bengaluru. Develop and maintain applications, review code, and test releases. "
          "Requires a Computer Science degree and Python experience. Apply on the official careers page.")


def test_real_model_returns_valid_output_and_ranks_scam_higher():
    service = distilbert_service.DistilBERTJobScamService()
    scam, normal = service.predict(SCAM), service.predict(NORMAL)
    for result in (scam, normal):
        assert 0.0 <= result["fraud_probability"] <= 1.0
        assert result["prediction"] in (0, 1)
    assert scam["fraud_probability"] > normal["fraud_probability"]
