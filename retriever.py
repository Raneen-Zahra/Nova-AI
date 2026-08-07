import json
import numpy as np
from vector_store import embed_text, EMBEDDINGS_FILE

TOP_K = 5


class Retriever:
    def __init__(self, top_k: int = TOP_K):
        self.top_k = top_k
        self.documents = self._load_index()

    def _load_index(self) -> list[dict]:
        try:
            with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print("[Retriever Warning]: No embeddings.json found. Run vector_store.py first.")
            return []

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        a = np.array(a)
        b = np.array(b)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        """
        Embeds the query and returns the top matching chunks,
        ranked by cosine similarity (highest first).
        """
        if not self.documents:
            return []

        k = top_k or self.top_k
        query_embedding = embed_text(query)

        scored = []
        for doc in self.documents:
            similarity = self._cosine_similarity(query_embedding, doc["embedding"])
            scored.append({
                "text": doc["text"],
                "metadata": {"source": doc["source"]},
                "distance": 1 - similarity,  # keep same "distance" semantics as before (lower = closer)
            })

        scored.sort(key=lambda x: x["distance"])
        return scored[:k]

    def format_context(self, retrieved_chunks: list[dict]) -> str:
        if not retrieved_chunks:
            return ""
        return "\n\n".join(
            f"[Chunk {i+1}]\n{chunk['text']}"
            for i, chunk in enumerate(retrieved_chunks)
        )


# --- Quick manual test ---
if __name__ == "__main__":
    retriever = Retriever(top_k=3)
    test_query = "What is this document about?"

    results = retriever.retrieve(test_query)
    print(f"Retrieved {len(results)} chunks for query: '{test_query}'\n")

    for i, r in enumerate(results):
        print(f"--- Chunk {i+1} (distance: {r['distance']:.4f}) ---")
        print(r["text"][:200], "...\n")

    print("=== Formatted context for prompt injection ===")
    print(retriever.format_context(results))