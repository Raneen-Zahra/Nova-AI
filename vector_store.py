import os
import json
from google import genai
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from document_loader import load_txt_documents

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from .env")

genai_client = genai.Client(api_key=api_key)

EMBEDDINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings.json")


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=20))
def embed_text(text: str) -> list[float]:
    """
    Embeds a single string using Google's Gemini embedding model.
    Retries automatically on transient network errors.
    """
    response = genai_client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=text,
    )
    return response.embeddings[0].values


def index_documents():
    """
    Loads text chunks from document_loader, embeds them via Gemini,
    and saves them (text + embedding + metadata) to a local JSON file.
    """
    processed_docs = load_txt_documents()

    if not processed_docs:
        print("[VectorStore Warning]: No documents available to index.")
        return 0

    print("[VectorStore]: Embedding text chunks via Google API...")

    indexed = []
    for doc in processed_docs:
        embedding = embed_text(doc["text"])
        indexed.append({
            "id": doc["id"],
            "text": doc["text"],
            "source": doc["source"],
            "embedding": embedding,
        })

    with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(indexed, f)

    print(f"[VectorStore Success]: Successfully indexed docs! Total vectors in DB: {len(indexed)}")
    return len(indexed)


if __name__ == "__main__":
    index_documents()