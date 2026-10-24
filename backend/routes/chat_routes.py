"""Flask blueprint for plain-text WhatsApp/Telegram chat analysis."""
from __future__ import annotations

from flask import Blueprint, jsonify, request
from werkzeug.exceptions import RequestEntityTooLarge

from backend.services.conversation_parser import parse_chat_text
from backend.services.conversation_analysis import analyze_messages

chat_bp = Blueprint("chat_routes", __name__)
MAX_CHAT_BYTES = 1 * 1024 * 1024  # 1 MiB


@chat_bp.errorhandler(RequestEntityTooLarge)
def too_large(_error):
    return jsonify({"error": "Chat file is too large. Please upload a .txt file under 1 MiB."}), 413


@chat_bp.post("/api/analyze-chat")
def analyze_chat_route():
    """Accept multipart/form-data with a .txt file in field 'file'."""
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"error": "Please upload a chat export as a .txt file."}), 400
    if not upload.filename.lower().endswith(".txt"):
        return jsonify({"error": "Unsupported file type. Please upload a plain-text .txt export."}), 400
    raw = upload.read(MAX_CHAT_BYTES + 1)
    if len(raw) > MAX_CHAT_BYTES:
        return jsonify({"error": "Chat file is too large. Please upload a .txt file under 1 MiB."}), 413
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return jsonify({"error": "Could not read this file as UTF-8 text. Re-export it as plain text."}), 400
    messages = parse_chat_text(text)
    if not messages:
        return jsonify({"error": "No supported timestamped messages were found in this file."}), 400
    # Do not persist the upload or include raw chat content in logs.
    return jsonify(analyze_messages(messages)), 200
