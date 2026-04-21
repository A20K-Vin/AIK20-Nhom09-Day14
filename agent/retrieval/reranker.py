from typing import List, Dict


class Reranker:
    """
    Score-based reranker: re-orders candidates by their vector similarity score
    and optionally boosts chunks that contain exact query terms.
    Override rerank() to plug in a cross-encoder model (e.g. sentence-transformers).
    """

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 3) -> List[Dict]:
        query_terms = set(query.lower().split())
        scored = []
        for c in candidates:
            keyword_boost = sum(1 for t in query_terms if t in c["text"].lower())
            combined = c["score"] + 0.02 * keyword_boost
            scored.append({**c, "rerank_score": combined})

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]
