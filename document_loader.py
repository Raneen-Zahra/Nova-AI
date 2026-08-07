# document_loader.py
import os
import glob
from typing import List, Dict

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """
    Splits text into overlapping sliding-window chunks to preserve semantic boundaries.
    """
    if not text.strip():
        return []
        
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # Move window forward by chunk_size MINUS overlap
        start += (chunk_size - overlap)

    return chunks

def load_txt_documents(folder_path: str = "./knowledge_base") -> List[Dict[str, str]]:
    """
    Reads all text files in the target directory and returns chunked data with metadata.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"[Mentor Note]: Created missing folder '{folder_path}'. Add your text files here!")
        return []

    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    if not txt_files:
        print(f"[Mentor Warning]: No .txt files found in '{folder_path}'.")
        return []

    processed_docs = []
    for file_path in txt_files:
        filename = os.path.basename(file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            chunks = chunk_text(content)
            for idx, chunk in enumerate(chunks):
                processed_docs.append({
                    "id": f"{filename}_chunk_{idx}",
                    "text": chunk,
                    "source": filename
                })
        except Exception as e:
            print(f"[Error reading {filename}]: {e}")

    return processed_docs

if __name__ == "__main__":
    # Quick sanity check run
    docs = load_txt_documents()
    print(f"Loaded {len(docs)} total chunks.")