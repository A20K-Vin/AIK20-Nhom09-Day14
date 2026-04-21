import json
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

try:
    import faiss
except ImportError as e:
    raise ImportError("faiss-cpu is required. Run: pip install faiss-cpu") from e


class VectorStore:
    def __init__(self, dim: int = 1536, persist_path: Optional[str] = None):
        self.dim = dim
        self.persist_path = persist_path
        # Inner-product index; vectors are L2-normalised before insertion → cosine similarity
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: List[Dict] = []

    def add(self, texts: List[str], embeddings: List[List[float]], metadata: List[Dict]):
        vecs = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(vecs)
        self.index.add(vecs)
        for idx, (text, meta) in enumerate(zip(texts, metadata)):
            # Đảm bảo mỗi chunk có chunk_id duy nhất
            chunk_id = meta.get("chunk_id") or meta.get("id") or f"chunk_{len(self.metadata)+1}"
            self.metadata.append({**meta, "text": text, "chunk_id": chunk_id})

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        vec = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(vec)
        scores, indices = self.index.search(vec, top_k)
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
        faiss.write_index(self.index, str(p / "index.faiss"))
        with open(p / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load(self):
        if not self.persist_path:
            return
        p = Path(self.persist_path)
        self.index = faiss.read_index(str(p / "index.faiss"))
        with open(p / "metadata.json", encoding="utf-8") as f:
            self.metadata = json.load(f)
