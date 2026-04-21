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
        # Placeholder logic
        total_hit_rate = 0
        total_mrr = 0
        for data in dataset:
            expected_ids = data['expected_retrieval_ids']
            retrieved_ids = data['retrieved_ids']
            total_hit_rate += self.calculate_hit_rate(expected_ids, retrieved_ids)
            total_mrr += self.calculate_mrr(expected_ids, retrieved_ids)
        avg_hit_rate = total_hit_rate / len(dataset)
        avg_mrr = total_mrr / len(dataset)
        return {"avg_hit_rate": avg_hit_rate, "avg_mrr": avg_mrr}
