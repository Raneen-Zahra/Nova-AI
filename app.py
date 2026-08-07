import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

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

# Allow your Netlify frontend to call this backend.
# Replace "*" with your actual Netlify URL once deployed, for security.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One shared engine instance (holds SQLite-backed history in memory).
# For a single-user personal assistant this is fine.
engine = GeminiEngine(system_instruction=SYSTEM_PROMPT)


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def health_check():
    return {"status": "NOVA is running"}


@app.post("/chat")
def chat(request: ChatRequest):
    def token_stream():
        for chunk in engine.stream_message(request.message):
            yield chunk

    return StreamingResponse(token_stream(), media_type="text/plain")


@app.post("/clear")
def clear_history():
    db.clear_history()
    engine.history = []
    return {"status": "history cleared"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)