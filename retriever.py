# retriever.py

import chromadb
from vector_store import GeminiEmbeddingFunction

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "my_rag_notes_v3"
TOP_K = 5


class Retriever:
    def __init__(self, top_k: int = TOP_K):
        self.top_k = top_k
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.embedding_fn = GeminiEmbeddingFunction()
        self.collection = self.client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn
        )

    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        k = top_k or self.top_k
        results = self.collection.query(query_texts=[query], n_results=k)
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        return [
            {"text": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(docs, metadatas, distances)
        ]

    def format_context(self, retrieved_chunks: list[dict]) -> str:
        if not retrieved_chunks:
            return ""
        return "\n\n".join(
            f"[Chunk {i+1}]\n{chunk['text']}"
            for i, chunk in enumerate(retrieved_chunks)
        )