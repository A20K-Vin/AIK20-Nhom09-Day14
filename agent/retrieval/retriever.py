from typing import List, Dict

from ingestion.embedder import Embedder
from retrieval.vector_store import VectorStore


class Retriever:
    def __init__(self, vector_store: VectorStore, embedder: Embedder, top_k: int = 5):
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k

    def search(self, query_embedding, top_k):
        import numpy as np

        query_vec = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for idx in indices[0]:
            results.append({
                "text": self.texts[idx],
                "chunk_id": self.metadatas[idx].get("id"),  
                "section": self.metadatas[idx].get("section"),
                "source": self.metadatas[idx].get("source"),
                "score": float(distances[0][len(results)])  # optional
            })
        return results

    def retrieve(self, query: str) -> List[Dict]:
        query_embedding = self.embedder.embed_one(query)
        return self.vector_store.search(query_embedding, self.top_k)
