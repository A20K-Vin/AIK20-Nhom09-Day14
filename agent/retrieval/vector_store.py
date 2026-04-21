import json
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None


class VectorStore:
    def __init__(self, dim: int = 1536, persist_path: Optional[str] = None):
        self.dim = dim
        self.persist_path = persist_path
        # Inner-product index; vectors are L2-normalised before insertion → cosine similarity
        self.index = faiss.IndexFlatIP(dim) if faiss else None
        self.metadata: List[Dict] = []
        self._vectors = np.empty((0, dim), dtype="float32")

    def add(self, texts: List[str], embeddings: List[List[float]], metadata: List[Dict]):
        vecs = np.array(embeddings, dtype="float32")
        if vecs.size == 0:
            return

        if faiss:
            faiss.normalize_L2(vecs)
        else:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms

        if self.index is not None:
            self.index.add(vecs)
        elif self._vectors.size == 0:
            self._vectors = vecs.copy()
        else:
            self._vectors = np.vstack([self._vectors, vecs])
        for idx, (text, meta) in enumerate(zip(texts, metadata)):
            # Đảm bảo mỗi chunk có chunk_id duy nhất
            chunk_id = meta.get("chunk_id") or meta.get("id") or f"chunk_{len(self.metadata)+1}"
            self.metadata.append({**meta, "text": text, "chunk_id": chunk_id})

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        if not self.metadata:
            return []

        vec = np.array([query_embedding], dtype="float32")
        if faiss:
            faiss.normalize_L2(vec)
        else:
            norm = np.linalg.norm(vec, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            vec = vec / norm

        if self.index is not None:
            scores, indices = self.index.search(vec, top_k)
        else:
            scores_1d = np.dot(self._vectors, vec[0])
            top_indices = np.argsort(scores_1d)[::-1][:top_k]
            indices = np.array([top_indices], dtype=int)
            scores = np.array([scores_1d[top_indices]], dtype="float32")
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            # Đảm bảo luôn có chunk_id
            chunk_id = meta.get("chunk_id") or meta.get("id") or f"chunk_{idx+1}"
            results.append({**meta, "chunk_id": chunk_id, "score": float(score)})
        return results

    def save(self):
        if not self.persist_path:
            return
        p = Path(self.persist_path)
        p.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(p / "index.faiss"))
        else:
            np.save(p / "vectors.npy", self._vectors)
        with open(p / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load(self):
        if not self.persist_path:
            return
        p = Path(self.persist_path)
        if self.index is not None:
            self.index = faiss.read_index(str(p / "index.faiss"))
        else:
            self._vectors = np.load(p / "vectors.npy")
        with open(p / "metadata.json", encoding="utf-8") as f:
            self.metadata = json.load(f)
