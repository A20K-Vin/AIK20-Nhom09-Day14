from typing import List, Dict

from ingestion.embedder import Embedder
from retrieval.vector_store import VectorStore


class Retriever:
    def __init__(self, vector_store: VectorStore, embedder: Embedder, top_k: int = 5):
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(self, query: str) -> List[Dict]:
        query_embedding = self.embedder.embed_one(query)
        return self.vector_store.search(query_embedding, self.top_k)
