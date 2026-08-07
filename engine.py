import os
import json
import requests
from dotenv import load_dotenv
import db
from retriever import Retriever

load_dotenv()

class GeminiEngine:
    def __init__(self, system_instruction: str, model: str = "gemini-3.1-flash-lite"):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY missing in .env")
        
        self.model = model
        self.endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:streamGenerateContent?alt=sse&key={self.api_key}"
        )
        self.base_system_instruction = system_instruction
        self.retriever = Retriever(top_k=5)

        db.init_db()
        self.history = db.load_recent_history(limit=10)

    def build_system_instruction(self, user_input: str) -> str:
        """Retrieves fresh context per query and merges it with the base instruction."""
        retrieved_chunks = self.retriever.retrieve(user_input)
        context = self.retriever.format_context(retrieved_chunks)

        if not context:
            context = "(No relevant documents found in the knowledge base.)"

        return (
            f"{self.base_system_instruction}\n\n"
            f"Use the following retrieved context to answer accurately. "
            f"If it's not relevant, ignore it and say you don't have that information.\n\n"
            f"--- RETRIEVED CONTEXT ---\n{context}\n--- END CONTEXT ---"
        )

    def stream_message(self, user_input: str):
        db.save_message("user", user_input)
        self.history.append({"role": "user", "parts": [{"text": user_input}]})

        recent_history = self.history[-10:]
        if recent_history and recent_history[0]["role"] != "user":
            recent_history = recent_history[1:]

        # Build fresh, query-specific system instruction with RAG context
        dynamic_system_instruction = self.build_system_instruction(user_input)

        payload = {
            "system_instruction": {"parts": [{"text": dynamic_system_instruction}]},
            "contents": recent_history,
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 400
            }
        }

        headers = {"Content-Type": "application/json"}
        full_response_text = ""

        try:
            with requests.post(self.endpoint, json=payload, headers=headers, stream=True) as response:
                if response.status_code != 200:
                    self.history.pop()
                    yield f"\n[API Error {response.status_code}]: {response.text}"
                    return

                for line in response.iter_lines():
                    if not line:
                        continue
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        json_str = line_str[6:].strip()
                        try:
                            data = json.loads(json_str)
                            candidates = data.get("candidates", [])
                            if candidates:
                                parts = candidates[0].get("content", {}).get("parts", [])
                                for part in parts:
                                    text_chunk = part.get("text", "")
                                    full_response_text += text_chunk
                                    yield text_chunk
                        except json.JSONDecodeError:
                            continue

            if full_response_text:
                db.save_message("model", full_response_text)
                self.history.append({"role": "model", "parts": [{"text": full_response_text}]})
            else:
                self.history.pop()

        except Exception as e:
            if self.history:
                self.history.pop()
            yield f"\n[STREAM ERROR]: {e}"