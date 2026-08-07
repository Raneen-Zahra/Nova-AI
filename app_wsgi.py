import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from a2wsgi import ASGIMiddleware

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

app = FastAPI(title="NOVA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = GeminiEngine(system_instruction=SYSTEM_PROMPT)


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def health_check():
    return {"status": "NOVA is running"}


@app.post("/chat")
def chat(request: ChatRequest):
    # WSGI hosts (like PythonAnywhere's free tier) buffer responses,
    # so we collect the full reply here instead of streaming it token-by-token.
    full_text = ""
    for chunk in engine.stream_message(request.message):
        full_text += chunk

    return {"response": full_text}


@app.post("/clear")
def clear_history():
    db.clear_history()
    engine.history = []
    return {"status": "history cleared"}


# PythonAnywhere's WSGI server needs a WSGI-callable named `application`.
# a2wsgi bridges our ASGI FastAPI app into a WSGI-compatible one.
application = ASGIMiddleware(app)