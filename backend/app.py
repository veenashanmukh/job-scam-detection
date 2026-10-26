
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

from backend.routes.chat_routes import chat_bp
from backend.services.llm_analysis import analyze


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2 MB
    CORS(app)

    app.register_blueprint(chat_bp)

    @app.errorhandler(RequestEntityTooLarge)
    def handle_request_too_large(error):
        return jsonify({"error": "Request payload is too large."}), 413

    @app.post("/api/analyze-job")
    def analyze_job_route():
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({"error": "Send job-posting text as JSON."}), 400

        text = data.get("text")

        if not isinstance(text, str) or not text.strip():
            return jsonify({"error": "Job-posting text cannot be empty."}), 400

        try:
            return jsonify(analyze(text)), 200
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 503

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"}), 200

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
