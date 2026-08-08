import os
import json

import db
from engine import GeminiEngine

SYSTEM_PROMPT = """
You are NOVA, an intelligent, warm, and highly capable personal AI assistant.

Core Persona & Guidelines:
1. Friendly & Grounded: Speak naturally, warmly, and encouragingly.
2. Structured & Clear: Deliver answers with maximum clarity. Use concise bullet points and bold key terms.
3. Solution-Oriented: Focus directly on resolving the user's intent immediately.
4. Actionable & Helpful: Answer the core intent first, then offer next steps.
5. Identity: If asked who created you, who you belong to, or who built you, mention that Raneen built you as a personal RAG-based AI assistant. Don't over-explain unless asked for details.
"""

engine = GeminiEngine(system_instruction=SYSTEM_PROMPT)


def _json_response(status, payload):
    body = json.dumps(payload).encode("utf-8")
    headers = [
        ("Content-Type", "application/json"),
        ("Content-Length", str(len(body))),
        ("Access-Control-Allow-Origin", "*"),
        ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
        ("Access-Control-Allow-Headers", "Content-Type"),
    ]
    return status, headers, [body]


def application(environ, start_response):
    method = environ.get("REQUEST_METHOD", "GET")
    path = environ.get("PATH_INFO", "/")

    if method == "OPTIONS":
        status, headers, body = _json_response("200 OK", {})
        start_response(status, headers)
        return body

    if path == "/" and method == "GET":
        status, headers, body = _json_response("200 OK", {"status": "NOVA is running"})
        start_response(status, headers)
        return body

    if path == "/chat" and method == "POST":
        try:
            content_length = int(environ.get("CONTENT_LENGTH", 0) or 0)
            raw_body = environ["wsgi.input"].read(content_length) if content_length else b"{}"
            data = json.loads(raw_body.decode("utf-8"))
            user_message = data.get("message", "").strip()

            if not user_message:
                status, headers, body = _json_response("400 Bad Request", {"error": "Missing 'message' field"})
                start_response(status, headers)
                return body

            full_text = ""
            for chunk in engine.stream_message(user_message):
                full_text += chunk

            status, headers, body = _json_response("200 OK", {"response": full_text})
            start_response(status, headers)
            return body

        except Exception as e:
            status, headers, body = _json_response("500 Internal Server Error", {"error": str(e)})
            start_response(status, headers)
            return body

    if path == "/clear" and method == "POST":
        db.clear_history()
        engine.history = []
        status, headers, body = _json_response("200 OK", {"status": "history cleared"})
        start_response(status, headers)
        return body

    status, headers, body = _json_response("404 Not Found", {"error": "Not found"})
    start_response(status, headers)
    return body