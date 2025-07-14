import chromadb
from embedder import Embedder
import hashlib


class RAGIndexer:
    def __init__(self, chroma_dir="./src/llm/chroma_store"):
        self.chroma_dir = chroma_dir
        self.chroma = chromadb.PersistentClient(path=chroma_dir)
        self.embedder = Embedder(model_dir="./models/mobilebert-onnx", use_cpu=False)

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def index_all(self, group, data):
        content = data
        doc_id = self._hash(content)

        collection = self.chroma.get_or_create_collection(name=group)
        existing = collection.get(ids=[doc_id])["ids"]
        if doc_id not in existing:
            print("Doc not found in collection, indexing...")
            vector = self.embedder.embed(content)[0]
            collection.add(documents=[content], embeddings=[vector], ids=[doc_id])
