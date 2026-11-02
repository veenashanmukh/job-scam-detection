"""fraud_logodds from the service, using a stub model so no weights are needed."""
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from backend.services.distilbert_service import DEFAULT_THRESHOLD, DistilBERTJobScamService


class _StubTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": torch.zeros((1, 3), dtype=torch.long)}


class _StubModel:
    def __init__(self, logits):
        self.logits = torch.tensor([logits], dtype=torch.float32)

    def __call__(self, **inputs):
        return type("Output", (), {"logits": self.logits})()


def _service(logits):
    service = DistilBERTJobScamService.__new__(DistilBERTJobScamService)
    service.device = torch.device("cpu")
    service.threshold = DEFAULT_THRESHOLD
    service.tokenizer = _StubTokenizer()
    service.model = _StubModel(logits)
    return service


def test_logodds_is_logit_difference_and_monotonic_with_probability():
    results = [_service([0.5, x / 4]).predict("posting") for x in range(-40, 41)]
    for low, high in zip(results, results[1:]):
        assert high["fraud_logodds"] > low["fraud_logodds"]
        assert high["fraud_probability"] >= low["fraud_probability"]
    for x, result in zip(range(-40, 41), results):
        assert result["fraud_logodds"] == pytest.approx(x / 4 - 0.5, abs=1e-6)
        assert 1 / (1 + math.exp(-result["fraud_logodds"])) == pytest.approx(result["fraud_probability"], abs=1e-6)


def test_logodds_is_not_rounded():
    result = _service([0.0, 1.234567891]).predict("posting")
    assert result["fraud_logodds"] != round(result["fraud_logodds"], 6)
