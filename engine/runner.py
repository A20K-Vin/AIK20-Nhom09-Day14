import asyncio
import time
from typing import Dict, List


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


class BenchmarkRunner:
    def __init__(
        self,
        agent,
        evaluator,
        judge,
        max_concurrency: int = 8,
        timeout_seconds: float = 45.0,
        max_retries: int = 1,
    ):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge
        self.max_concurrency = max(1, max_concurrency)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, max_retries)
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self._estimated_price_per_1k_tokens = {
            "agent_generation": 0.0015,
            "judge_models": 0.0030,
        }

    async def run_single_test(self, test_case: Dict) -> Dict:
        async with self._semaphore:
            started_at = time.perf_counter()
            question = test_case["question"]
            expected_answer = test_case["expected_answer"]
            last_error = None

            for attempt in range(self.max_retries + 1):
                try:
                    agent_started = time.perf_counter()
                    response = await asyncio.wait_for(
                        self.agent.query(question),
                        timeout=self.timeout_seconds,
                    )
                    agent_latency = time.perf_counter() - agent_started

                    eval_started = time.perf_counter()
                    ragas_scores = await asyncio.wait_for(
                        self.evaluator.score(test_case, response),
                        timeout=self.timeout_seconds,
                    )
                    eval_latency = time.perf_counter() - eval_started

                    judge_started = time.perf_counter()
                    judge_result = await asyncio.wait_for(
                        self.judge.evaluate_multi_judge(
                            question,
                            response["answer"],
                            expected_answer,
                        ),
                        timeout=self.timeout_seconds,
                    )
                    judge_latency = time.perf_counter() - judge_started
                    total_latency = time.perf_counter() - started_at

                    answer = response["answer"]
                    contexts = response.get("contexts", [])
                    context_texts = [
                        ctx.get("text", "") if isinstance(ctx, dict) else str(ctx)
                        for ctx in contexts
                    ]
                    context_text = "\n".join(context_texts)
                    agent_usage = response.get("metadata", {}).get("usage", {})
                    agent_prompt_tokens = int(agent_usage.get("prompt_tokens", 0))
                    agent_completion_tokens = int(agent_usage.get("completion_tokens", 0))
                    agent_tokens = int(agent_usage.get("total_tokens", 0))
                    if agent_tokens <= 0:
                        agent_tokens = _estimate_tokens(question + answer + context_text)
                    judge_tokens = int(judge_result.get("cost_tokens", 0))
                    total_tokens = agent_tokens + judge_tokens
                    estimated_cost_usd = (
                        (agent_tokens / 1000.0) * self._estimated_price_per_1k_tokens["agent_generation"]
                        + (judge_tokens / 1000.0) * self._estimated_price_per_1k_tokens["judge_models"]
                    )

                    return {
                        "test_case": question,
                        "expected_answer": expected_answer,
                        "reference_context": test_case.get("context", ""),
                        "case_metadata": test_case.get("metadata", {}),
                        "agent_response": answer,
                        "retrieved_contexts": context_texts,
                        "retrieved_ids": response.get("metadata", {}).get("retrieved_ids", []),
                        "retrieved_sources": response.get("metadata", {}).get("sources", []),
                        "latency": total_latency,
                        "latency_breakdown": {
                            "agent_seconds": agent_latency,
                            "evaluator_seconds": eval_latency,
                            "judge_seconds": judge_latency,
                            "total_seconds": total_latency,
                        },
                        "token_usage": {
                            "agent_prompt_tokens": agent_prompt_tokens,
                            "agent_completion_tokens": agent_completion_tokens,
                            "agent_total_tokens": agent_tokens,
                            "agent_usage_source": "api_usage" if int(agent_usage.get("total_tokens", 0)) > 0 else "estimated",
                            "judge_tokens": judge_tokens,
                            "total_tokens": total_tokens,
                        },
                        "cost_usd_estimated": estimated_cost_usd,
                        "ragas": ragas_scores,
                        "judge": judge_result,
                        "attempt": attempt + 1,
                        "status": "fail" if judge_result["final_score"] < 3 else "pass",
                    }
                except Exception as exc:
                    last_error = str(exc)
                    if attempt < self.max_retries:
                        await asyncio.sleep(1.0 * (attempt + 1))
                        continue

            total_latency = time.perf_counter() - started_at
            return {
                "test_case": question,
                "expected_answer": expected_answer,
                "reference_context": test_case.get("context", ""),
                "case_metadata": test_case.get("metadata", {}),
                "agent_response": "",
                "retrieved_contexts": [],
                "retrieved_ids": [],
                "retrieved_sources": [],
                "latency": total_latency,
                "latency_breakdown": {
                    "agent_seconds": None,
                    "evaluator_seconds": None,
                    "judge_seconds": None,
                    "total_seconds": total_latency,
                },
                "token_usage": {
                    "agent_prompt_tokens": 0,
                    "agent_completion_tokens": 0,
                    "agent_total_tokens": 0,
                    "agent_usage_source": "none",
                    "judge_tokens": 0,
                    "total_tokens": 0,
                },
                "cost_usd_estimated": 0.0,
                "ragas": {"faithfulness": 0.0, "relevancy": 0.0, "retrieval": {"hit_rate": 0.0, "mrr": 0.0}},
                "judge": {"final_score": 1.0, "agreement_rate": 0.0, "cost_tokens": 0},
                "attempt": self.max_retries + 1,
                "status": "error",
                "error": last_error or "unknown_error",
            }

    async def run_all(self, dataset: List[Dict]) -> List[Dict]:
        tasks = [asyncio.create_task(self.run_single_test(case)) for case in dataset]
        total = len(tasks)
        results = []
        for completed_count, task in enumerate(asyncio.as_completed(tasks), start=1):
            result = await task
            results.append(result)
            status = result.get("status", "unknown").upper()
            score = result.get("judge", {}).get("final_score", 0.0)
            error_detail = ""
            if status == "ERROR" and result.get("error"):
                short_error = str(result["error"]).replace("\n", " ").strip()
                error_detail = f" | error={short_error[:120]}"
            print(
                f"[Benchmark] {completed_count}/{total} | {status} | score={score:.2f} | {result.get('test_case', '')}{error_detail}",
                flush=True,
            )
        return results
