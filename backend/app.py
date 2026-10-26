"""Flask application factory.

Run from the repository root:  python -m backend.app
"""
from __future__ import annotations

from flask import Flask, jsonify
from flask_cors import CORS

from backend.routes.chat_routes import chat_bp
from backend.routes.job_routes import job_bp


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2 MiB hard cap on any request body
    CORS(app, resources={r"/api/*": {"origins": ["http://localhost:5173", "http://localhost:5174", "http://127.0.0.1:5173"]}})

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify({"error": "The request is too large."}), 413

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    app.register_blueprint(chat_bp)
    app.register_blueprint(job_bp)
    return app


if __name__ == "__main__":
    create_app().run(debug=True)
