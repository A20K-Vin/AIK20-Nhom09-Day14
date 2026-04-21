import asyncio
from collections import Counter
import json
import os
import time
from dotenv import load_dotenv
from engine.runner import BenchmarkRunner
from engine.llm_judge import LLMJudge
from engine.retrieval_eval import RetrievalEvaluator
from agent.main_agent import MainAgent

load_dotenv()

AGENT_PROFILES = {
    "Agent_V1_Base": {
        "model": "gpt-4o-mini",
        "top_k": 3,
        "rerank_top_k": 2,
        "use_reranker": False,
        "notes": "Baseline retrieval without reranking; narrower context budget.",
    },
    "Agent_V2_Optimized": {
        "model": "gpt-4o-mini",
        "top_k": 5,
        "rerank_top_k": 3,
        "use_reranker": True,
        "notes": "Optimized retrieval with reranking, latency/token tracking, and calibrated reporting.",
    },
}


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

def get_agent_profile(agent_version: str) -> dict:
    return AGENT_PROFILES.get(agent_version, AGENT_PROFILES["Agent_V2_Optimized"]).copy()

def build_agent(agent_version: str) -> MainAgent:
    profile = get_agent_profile(agent_version)
    return MainAgent(
        model=profile["model"],
        top_k=profile["top_k"],
        rerank_top_k=profile["rerank_top_k"],
        use_reranker=profile["use_reranker"],
    )

def build_release_gate(v1_summary: dict, v2_summary: dict) -> dict:
    v1_metrics = v1_summary["metrics"]
    v2_metrics = v2_summary["metrics"]
    score_delta = v2_metrics["avg_score"] - v1_metrics["avg_score"]
    hit_rate_delta = v2_metrics["hit_rate"] - v1_metrics["hit_rate"]
    pass_rate_delta = v2_metrics["pass_rate"] - v1_metrics["pass_rate"]
    cost_delta = v2_metrics["avg_estimated_cost_per_case_usd"] - v1_metrics["avg_estimated_cost_per_case_usd"]

    checks = {
        "score_not_worse": score_delta >= 0.0,
        "pass_rate_floor": v2_metrics["pass_rate"] >= 0.80,
        "agreement_floor": v2_metrics["agreement_rate"] >= 0.75,
        "retrieval_not_worse": hit_rate_delta >= -0.05,
        "error_rate_floor": v2_metrics["error_rate"] <= 0.05,
        "cost_budget": v2_metrics["avg_estimated_cost_per_case_usd"] <= 0.01,
    }
    decision = "APPROVE" if all(checks.values()) else "BLOCK_RELEASE"
    return {
        "decision": decision,
        "checks": checks,
        "deltas": {
            "avg_score": score_delta,
            "hit_rate": hit_rate_delta,
            "pass_rate": pass_rate_delta,
            "avg_estimated_cost_per_case_usd": cost_delta,
        },
    }

def build_failure_analysis(results: list, summary: dict, regression: dict) -> str:
    total = len(results)
    passed_count = sum(1 for r in results if r["status"] == "pass")
    failed = [r for r in results if r["status"] != "pass"]
    fail_counter = Counter(r.get("case_metadata", {}).get("type", "unknown") for r in failed)
    cluster_lines = []
    if fail_counter:
        for failure_type, count in fail_counter.most_common(5):
            sample = next(
                (r for r in failed if r.get("case_metadata", {}).get("type", "unknown") == failure_type),
                None,
            )
            retrieval_mode = sample.get("ragas", {}).get("retrieval", {}).get("evaluation_mode", "unknown") if sample else "unknown"
            cluster_lines.append(
                f"| {failure_type} | {count} | Retrieval mode: {retrieval_mode}; needs tighter context selection or stronger refusal behavior. |"
            )
    else:
        cluster_lines.append("| Không có fail case | 0 | Tất cả case đều pass theo ngưỡng judge hiện tại. |")

    worst_cases = sorted(
        results,
        key=lambda r: (
            r.get("judge", {}).get("final_score", 0.0),
            r.get("ragas", {}).get("faithfulness", 0.0),
            -r.get("latency", 0.0),
        ),
    )[:3]

    case_sections = []
    for idx, case in enumerate(worst_cases, start=1):
        metadata = case.get("case_metadata", {})
        retrieval = case.get("ragas", {}).get("retrieval", {})
        case_sections.append(
            "\n".join(
                [
                    f"### Case #{idx}: {case['test_case']}",
                    f"1. **Symptom:** Judge score {case.get('judge', {}).get('final_score', 0.0):.2f}/5, faithfulness {case.get('ragas', {}).get('faithfulness', 0.0):.2f}, retrieval hit rate {retrieval.get('hit_rate', 0.0):.2f}.",
                    f"2. **Why 1:** Câu hỏi thuộc nhóm `{metadata.get('type', 'unknown')}` với độ khó `{metadata.get('difficulty', 'unknown')}`, nên rất nhạy với context sai hoặc thiếu.",
                    f"3. **Why 2:** Retrieval đang chạy ở chế độ `{retrieval.get('evaluation_mode', 'unknown')}`; nếu không hit ở rank đầu, answer quality giảm rõ rệt.",
                    f"4. **Why 3:** Agent trả về {len(case.get('retrieved_contexts', []))} context, nên nhiễu retrieval hoặc chunk quá rộng vẫn có thể kéo tụt precision.",
                    f"5. **Why 4:** Judge agreement = {case.get('judge', {}).get('agreement_rate', 0.0):.2f}; khi agreement chưa tuyệt đối, câu trả lời thường đúng một phần nhưng chưa bám sát policy wording.",
                    "6. **Root Cause:** Cần tiếp tục tối ưu mapping từ câu hỏi khó sang chunk giàu tín hiệu hơn, đồng thời siết cách agent trả lời cho các case adversarial/ambiguous.",
                ]
            )
        )

    metrics = summary["metrics"]
    return "\n".join(
        [
            "# Báo cáo Phân tích Thất bại (Failure Analysis Report)",
            "",
            "## 1. Tổng quan Benchmark",
            f"- **Tổng số cases:** {summary['metadata']['total']}",
            f"- **Tỉ lệ Pass/Fail:** {passed_count}/{total - passed_count}",
            "- **Điểm RAGAS trung bình:**",
            f"  - Faithfulness: {metrics['avg_faithfulness']:.4f}",
            f"  - Relevancy: {metrics['avg_relevancy']:.4f}",
            f"  - Hit Rate: {metrics['hit_rate']:.4f}",
            f"  - MRR: {metrics['mrr']:.4f}",
            f"- **Điểm LLM-Judge trung bình:** {metrics['avg_score']:.2f} / 5.0",
            f"- **Agreement Rate:** {metrics['agreement_rate']:.2%}",
            f"- **Batch Cohen's Kappa:** {metrics['batch_cohens_kappa']:.4f}",
            f"- **Avg latency / case:** {metrics['avg_latency_seconds']:.2f}s",
            f"- **Avg estimated cost / case:** ${metrics['avg_estimated_cost_per_case_usd']:.4f}",
            "",
            "## 2. Phân nhóm lỗi (Failure Clustering)",
            "| Nhóm lỗi | Số lượng | Nguyên nhân dự kiến |",
            "|----------|----------|---------------------|",
            *cluster_lines,
            "",
            "## 3. Phân tích 5 Whys (Chọn 3 case tệ nhất)",
            "",
            *case_sections,
            "",
            "## 4. Kế hoạch cải tiến (Action Plan)",
            f"- [x] Đồng bộ benchmark để chạy từ project root và ghi đầy đủ latency/token/cost vào report.",
            f"- [x] Bổ sung fallback retrieval evaluation bằng context overlap khi curated `ground_truth_ids` không cùng namespace với runtime chunk IDs.",
            f"- [x] Thiết lập regression gate với quyết định hiện tại: **{regression['decision']}**.",
            "- [ ] Chuẩn hóa lại hệ chunk ID trong vector store để khớp 1-1 với curated ground-truth IDs của dataset.",
            "- [ ] Tạo thêm một baseline agent riêng biệt ở code-level để regression phản ánh chính xác từng thay đổi kiến trúc hơn nữa.",
        ]
    )

def validate_runtime_requirements() -> bool:
    missing = []
    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")

    if missing:
        print("❌ Thiếu biến môi trường bắt buộc để chạy benchmark thật:")
        for key in missing:
            print(f"   - {key}")
        print("💡 Hãy thêm API key vào .env hoặc environment hiện tại rồi chạy lại.")
        return False

    return True

async def run_benchmark_with_results(agent_version: str):
    print(f"🚀 Khởi động Benchmark cho {agent_version}...")

    if not validate_runtime_requirements():
        return None, None

    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("❌ File data/golden_set.jsonl rỗng. Hãy tạo ít nhất 1 test case.")
        return None, None

    runner = BenchmarkRunner(
        build_agent(agent_version),
        ExpertEvaluator(),
        MultiModelJudge(),
        max_concurrency=4,
        timeout_seconds=45.0,
        max_retries=1,
    )
    print(
        f"   ↳ Profile: top_k={get_agent_profile(agent_version)['top_k']}, "
        f"rerank_top_k={get_agent_profile(agent_version)['rerank_top_k']}, "
        f"use_reranker={get_agent_profile(agent_version)['use_reranker']}",
        flush=True,
    )
    results = await runner.run_all(dataset)
    print(f"✅ Hoàn tất {agent_version}: {len(results)} cases", flush=True)

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
            "agent_profile": get_agent_profile(agent_version),
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
    v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized")
    
    if not v1_summary or not v2_summary:
        print("❌ Không thể chạy Benchmark. Kiểm tra lại data/golden_set.jsonl.")
        return

    regression = build_release_gate(v1_summary, v2_summary)
    v2_summary["regression"] = {
        "baseline_version": v1_summary["metadata"]["version"],
        "candidate_version": v2_summary["metadata"]["version"],
        **regression,
    }

    print("\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    delta = regression["deltas"]["avg_score"]
    print(f"V1 Score: {v1_summary['metrics']['avg_score']:.2f}")
    print(f"V2 Score: {v2_summary['metrics']['avg_score']:.2f}")
    print(f"Delta: {'+' if delta >= 0 else ''}{delta:.2f}")
    print(f"Release Gate: {regression['decision']}")

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)
    with open("analysis/failure_analysis.md", "w", encoding="utf-8") as f:
        f.write(build_failure_analysis(v2_results, v2_summary, regression))

    if regression["decision"] == "APPROVE":
        print("✅ QUYẾT ĐỊNH: CHẤP NHẬN BẢN CẬP NHẬT (APPROVE)")
    else:
        print("❌ QUYẾT ĐỊNH: TỪ CHỐI (BLOCK RELEASE)")

if __name__ == "__main__":
    asyncio.run(main())
