from typing import List, Dict

class RetrievalEvaluator:
    def __init__(self):
        pass

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        """
        TODO: Tính toán xem ít nhất 1 trong expected_ids có nằm trong top_k của retrieved_ids không.
        """
        top_retrieved = retrieved_ids[:top_k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        TODO: Tính Mean Reciprocal Rank.
        Tìm vị trí đầu tiên của một expected_id trong retrieved_ids.
        MRR = 1 / position (vị trí 1-indexed). Nếu không thấy thì là 0.
        """
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    async def evaluate_batch(self, dataset: List[Dict]) -> Dict:
        """
        Chạy eval cho toàn bộ bộ dữ liệu.
        Dataset cần có trường 'expected_retrieval_ids' và Agent trả về 'retrieved_ids'.
        """
        if not dataset:
            return {"avg_hit_rate": 0.0, "avg_mrr": 0.0}

        hit_rates = []
        mrr_scores = []
        for item in dataset:
            expected_ids = item.get("expected_retrieval_ids", [])
            retrieved_ids = item.get("retrieved_ids", [])
            if expected_ids and retrieved_ids:
                hit_rates.append(self.calculate_hit_rate(expected_ids, retrieved_ids))
                mrr_scores.append(self.calculate_mrr(expected_ids, retrieved_ids))
                continue

            expected_context = item.get("expected_context", "")
            retrieved_contexts = item.get("retrieved_contexts", [])
            hit_rates.append(self.calculate_context_hit_rate(expected_context, retrieved_contexts))
            mrr_scores.append(self.calculate_context_mrr(expected_context, retrieved_contexts))

        return {
            "avg_hit_rate": sum(hit_rates) / len(hit_rates),
            "avg_mrr": sum(mrr_scores) / len(mrr_scores),
        }

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join((text or "").lower().split())

    def calculate_context_hit_rate(
        self, expected_context: str, retrieved_contexts: List[str], overlap_threshold: float = 0.35
    ) -> float:
        """
        Fallback retrieval quality when no doc IDs are available.
        We compute token-overlap between expected context and each retrieved context.
        Hit = 1 if at least one overlap ratio >= threshold.
        """
        expected_tokens = set(self._normalize_text(expected_context).split())
        if not expected_tokens:
            return 0.0

        for ctx in retrieved_contexts:
            ctx_tokens = set(self._normalize_text(ctx).split())
            if not ctx_tokens:
                continue
            overlap = len(expected_tokens.intersection(ctx_tokens)) / len(expected_tokens)
            if overlap >= overlap_threshold:
                return 1.0
        return 0.0

    def calculate_context_mrr(
        self, expected_context: str, retrieved_contexts: List[str], overlap_threshold: float = 0.35
    ) -> float:
        expected_tokens = set(self._normalize_text(expected_context).split())
        if not expected_tokens:
            return 0.0

        for i, ctx in enumerate(retrieved_contexts):
            ctx_tokens = set(self._normalize_text(ctx).split())
            if not ctx_tokens:
                continue
            overlap = len(expected_tokens.intersection(ctx_tokens)) / len(expected_tokens)
            if overlap >= overlap_threshold:
                return 1.0 / (i + 1)
        return 0.0

    def evaluate_case(self, test_case: Dict, response: Dict, top_k: int = 3) -> Dict:
        metadata = test_case.get("metadata", {})
        expected_ids = metadata.get("ground_truth_ids") or test_case.get("expected_retrieval_ids", [])
        retrieved_ids = response.get("metadata", {}).get("retrieved_ids", [])

        if expected_ids and retrieved_ids:
            return {
                "hit_rate": self.calculate_hit_rate(expected_ids, retrieved_ids, top_k=top_k),
                "mrr": self.calculate_mrr(expected_ids, retrieved_ids),
                "evaluation_mode": "ground_truth_ids",
            }

        expected_context = test_case.get("context", "")
        retrieved_contexts = response.get("contexts", [])
        return {
            "hit_rate": self.calculate_context_hit_rate(expected_context, retrieved_contexts),
            "mrr": self.calculate_context_mrr(expected_context, retrieved_contexts),
            "evaluation_mode": "context_overlap",
        }
