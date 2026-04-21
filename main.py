import asyncio
import json
import os
import time
from engine.runner import BenchmarkRunner
from engine.llm_judge import LLMJudge
from engine.retrieval_eval import RetrievalEvaluator
from agent.main_agent import MainAgent
from engine.retrieval_eval import RetrievalEvaluator


class ExpertEvaluator:
    def __init__(self):
        self.retrieval_eval = RetrievalEvaluator()

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join((text or "").lower().split())

    @classmethod
    def _token_overlap_ratio(cls, reference: str, candidate: str) -> float:
        ref_tokens = set(cls._normalize_text(reference).split())
        cand_tokens = set(cls._normalize_text(candidate).split())
        if not ref_tokens:
            return 0.0
        return len(ref_tokens.intersection(cand_tokens)) / len(ref_tokens)

    async def score(self, case, resp):
        answer = resp.get("answer", "")
        expected_context = case.get("context", "")
        expected_answer = case.get("expected_answer", "")
        retrieval_scores = self.retrieval_eval.evaluate_case(case, resp)

        # Lightweight lexical proxy metrics (stable, no extra API cost)
        faithfulness = self._token_overlap_ratio(expected_context, answer)
        relevancy = self._token_overlap_ratio(expected_answer, answer)

        return {
            "faithfulness": round(min(1.0, faithfulness), 4),
            "relevancy": round(min(1.0, relevancy), 4),
            "retrieval": retrieval_scores,
        }

# Dùng LLMJudge thực thay cho placeholder
MultiModelJudge = LLMJudge

async def run_benchmark_with_results(agent_version: str):
    print(f"🚀 Khởi động Benchmark cho {agent_version}...")

    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("❌ File data/golden_set.jsonl rỗng. Hãy tạo ít nhất 1 test case.")
        return None, None

    runner = BenchmarkRunner(
        MainAgent(),
        ExpertEvaluator(),
        MultiModelJudge(),
        max_concurrency=8,
        timeout_seconds=45.0,
        max_retries=1,
    )
    results = await runner.run_all(dataset)

    total = len(results)
    passed = [r for r in results if r["status"] == "pass"]
    non_error = [r for r in results if r["status"] != "error"]
    avg_latency = sum(r["latency"] for r in non_error) / len(non_error) if non_error else 0.0
    p95_latency = (
        sorted(r["latency"] for r in non_error)[max(0, int(0.95 * len(non_error)) - 1)]
        if non_error
        else 0.0
    )
    total_tokens = sum(r.get("token_usage", {}).get("total_tokens", 0) for r in results)
    total_agent_tokens = sum(r.get("token_usage", {}).get("agent_total_tokens", 0) for r in results)
    total_judge_tokens = sum(r.get("token_usage", {}).get("judge_tokens", 0) for r in results)
    total_cost = sum(r.get("cost_usd_estimated", 0.0) for r in results)
    judge_pairs_a = []
    judge_pairs_b = []
    for result in results:
        individual_scores = result.get("judge", {}).get("individual_scores", {})
        secondary_judges = [key for key in individual_scores.keys() if key != "gpt-4o"]
        if "gpt-4o" in individual_scores and secondary_judges:
            secondary_name = secondary_judges[0]
            judge_pairs_a.append(individual_scores["gpt-4o"])
            judge_pairs_b.append(individual_scores[secondary_name])
    batch_kappa = (
        LLMJudge.calculate_batch_kappa(judge_pairs_a, judge_pairs_b)
        if judge_pairs_a and judge_pairs_b
        else 0.0
    )

    summary = {
        "metadata": {
            "version": agent_version,
            "total": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": {
            "avg_score": sum(r["judge"]["final_score"] for r in results) / total,
            "hit_rate": sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total,
            "mrr": sum(r["ragas"]["retrieval"]["mrr"] for r in results) / total,
            "agreement_rate": sum(r["judge"]["agreement_rate"] for r in results) / total,
            "batch_cohens_kappa": batch_kappa,
            "avg_faithfulness": sum(r["ragas"]["faithfulness"] for r in results) / total,
            "avg_relevancy": sum(r["ragas"]["relevancy"] for r in results) / total,
            "pass_rate": len(passed) / total,
            "avg_latency_seconds": avg_latency,
            "p95_latency_seconds": p95_latency,
            "total_tokens": total_tokens,
            "total_agent_tokens": total_agent_tokens,
            "total_judge_tokens": total_judge_tokens,
            "avg_tokens_per_case": total_tokens / total if total else 0.0,
            "avg_agent_tokens_per_case": total_agent_tokens / total if total else 0.0,
            "avg_judge_tokens_per_case": total_judge_tokens / total if total else 0.0,
            "total_estimated_cost_usd": total_cost,
            "avg_estimated_cost_per_case_usd": total_cost / total if total else 0.0,
            "error_rate": (total - len(non_error)) / total if total else 0.0,
        },
        "runner_config": {
            "max_concurrency": runner.max_concurrency,
            "timeout_seconds": runner.timeout_seconds,
            "max_retries": runner.max_retries,
        },
    }
    return results, summary

async def run_benchmark(version):
    _, summary = await run_benchmark_with_results(version)
    return summary

async def main():
    v1_summary = await run_benchmark("Agent_V1_Base")
    
    # Giả lập V2 có cải tiến (để test logic)
    v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized")
    
    if not v1_summary or not v2_summary:
        print("❌ Không thể chạy Benchmark. Kiểm tra lại data/golden_set.jsonl.")
        return

    print("\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    delta = v2_summary["metrics"]["avg_score"] - v1_summary["metrics"]["avg_score"]
    print(f"V1 Score: {v1_summary['metrics']['avg_score']}")
    print(f"V2 Score: {v2_summary['metrics']['avg_score']}")
    print(f"Delta: {'+' if delta >= 0 else ''}{delta:.2f}")

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)

    if delta > 0:
        print("✅ QUYẾT ĐỊNH: CHẤP NHẬN BẢN CẬP NHẬT (APPROVE)")
    else:
        print("❌ QUYẾT ĐỊNH: TỪ CHỐI (BLOCK RELEASE)")

if __name__ == "__main__":
    asyncio.run(main())
