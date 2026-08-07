# vector_store.py
import os
import chromadb
from google import genai
from dotenv import load_dotenv
from document_loader import load_txt_documents

load_dotenv()

# 1. Initialize Google GenAI client
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from .env")

genai_client = genai.Client(api_key=api_key)

# 2. Define custom embedding function complying with ChromaDB standards
from tenacity import retry, stop_after_attempt, wait_exponential

class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self):
        pass

    def name(self) -> str:
        return "gemini_embedding_function"

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=20))
    def _embed_one(self, doc: str):
        response = genai_client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=doc,
        )
        return response.embeddings[0].values

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        return [self._embed_one(doc) for doc in input]
# 3. Create persistent database client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Delete existing collection if present to avoid embedding function mismatch
try:
    chroma_client.delete_collection(name="my_rag_notes_v2")
except Exception:
    pass

# 4. Initialize collection with Gemini embedding function
collection = chroma_client.get_or_create_collection(
    name="my_rag_notes_v3",
    embedding_function=GeminiEmbeddingFunction()
)

def index_documents():
    """
    Loads text chunks from document_loader and indexes them into ChromaDB using Gemini embeddings.
    """
    processed_docs = load_txt_documents()
    
    if not processed_docs:
        print("[VectorStore Warning]: No documents available to index.")
        return 0

    ids = [doc["id"] for doc in processed_docs]
    texts = [doc["text"] for doc in processed_docs]
    metadatas = [{"source": doc["source"]} for doc in processed_docs]

    print("[VectorStore]: Embedding text chunks via Google API...")
    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas
    )

    total_count = collection.count()
    print(f"[VectorStore Success]: Successfully indexed docs! Total vectors in DB: {total_count}")
    return total_count

if __name__ == "__main__":
    index_documents()