import io
import pytest

flask = pytest.importorskip("flask")
from flask import Flask
from backend.routes.chat_routes import chat_bp


@pytest.fixture()
def client():
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(chat_bp)
    return app.test_client()


def test_chat_endpoint_accepts_txt_and_returns_report(client):
    response = client.post("/api/analyze-chat", data={
        "file": (io.BytesIO(b"24/10/26, 14:30 - Recruiter: Pay Rs 2500 fee today"), "chat.txt")
    }, content_type="multipart/form-data")
    assert response.status_code == 200
    body = response.get_json()
    assert "risk_score" in body and "timeline" in body
    assert "Payment request" in body["warning_signs"]


def test_chat_endpoint_rejects_wrong_extension(client):
    response = client.post("/api/analyze-chat", data={
        "file": (io.BytesIO(b"hello"), "chat.pdf")
    }, content_type="multipart/form-data")
    assert response.status_code == 400


def test_chat_endpoint_rejects_missing_file(client):
    response = client.post("/api/analyze-chat", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
