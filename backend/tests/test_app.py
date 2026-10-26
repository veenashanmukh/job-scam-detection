import pytest

pytest.importorskip("flask")
pytest.importorskip("flask_cors")
from backend.app import create_app


def test_health_and_routes_are_registered():
    client = create_app().test_client()
    assert client.get("/api/health").get_json() == {"status": "ok"}
    rules = {rule.rule for rule in create_app().url_map.iter_rules()}
    assert {"/api/analyze-job", "/api/analyze-chat", "/api/health"} <= rules


def test_oversized_request_returns_json_413():
    client = create_app().test_client()
    response = client.post("/api/analyze-job", data=b"x" * (3 * 1024 * 1024), content_type="application/json")
    assert response.status_code == 413
    assert "error" in response.get_json()
