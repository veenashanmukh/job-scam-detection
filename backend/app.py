
from flask import Flask, jsonify
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

from backend.routes.chat_routes import chat_bp
from backend.routes.job_routes import job_bp


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

    CORS(app)

    # Conversation analysis remains on its existing route.
    app.register_blueprint(chat_bp)

    # Job-posting analysis uses the existing DistilBERT classifier.
    app.register_blueprint(job_bp)

    @app.errorhandler(RequestEntityTooLarge)
    def handle_request_too_large(error):
        return jsonify({"error": "Request payload is too large."}), 413

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"}), 200

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
