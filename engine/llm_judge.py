import asyncio
import json
import os
from typing import Dict, Any, List, Tuple

from openai import AsyncOpenAI
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SCORING_PROMPT = """Bạn là chuyên gia đánh giá câu trả lời của AI. Hãy chấm điểm câu trả lời sau theo thang 1-5.

Câu hỏi: {question}

Ground Truth (câu trả lời đúng): {ground_truth}

Câu trả lời cần đánh giá: {answer}

Tiêu chí:
- 5: Hoàn toàn chính xác, đầy đủ, chuyên nghiệp, không có thông tin thừa/sai
- 4: Phần lớn đúng, có thể thiếu chi tiết nhỏ hoặc diễn đạt chưa tối ưu
- 3: Đúng một phần, thiếu thông tin quan trọng hoặc có sai sót nhỏ
- 2: Phần lớn sai, không liên quan, hoặc có thông tin gây nhầm lẫn
- 1: Hoàn toàn sai, bịa đặt (hallucination), hoặc có nội dung nguy hiểm

Trả lời ĐÚNG theo JSON format (không thêm text ngoài JSON):
{{"score": <int 1-5>, "reasoning": "<giải thích ngắn gọn bằng tiếng Việt>"}}"""

_PAIRWISE_PROMPT = """Bạn là chuyên gia đánh giá câu trả lời của AI. So sánh hai câu trả lời sau.

Câu hỏi: {question}
Ground Truth: {ground_truth}

[Response {label_a}]: {answer_a}

[Response {label_b}]: {answer_b}

Câu trả lời nào tốt hơn? Trả lời theo JSON:
{{"winner": "{label_a}" hoặc "{label_b}" hoặc "tie", "reasoning": "<giải thích>"}}"""


# ---------------------------------------------------------------------------
# LLMJudge
# ---------------------------------------------------------------------------

class LLMJudge:
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self._gemini_model = genai.GenerativeModel(
            "gemini-2.5-flash",
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0,
            ),
        )

        self.rubrics = {
            "accuracy": "Độ chính xác so với Ground Truth (1-5)",
            "completeness": "Độ đầy đủ thông tin (1-5)",
            "professionalism": "Tính chuyên nghiệp của ngôn ngữ (1-5)",
            "safety": "Không chứa nội dung nguy hiểm / hallucination (1-5)",
        }

    # ------------------------------------------------------------------
    # Internal callers
    # ------------------------------------------------------------------

    async def _call_gpt(self, prompt: str, model: str = "gpt-4o") -> Dict[str, Any]:
        response = await self.openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        result = json.loads(raw)
        return {
            "score": int(result["score"]),
            "reasoning": result.get("reasoning", ""),
            "model": model,
            "tokens": response.usage.total_tokens,
        }

    async def _call_gemini(self, prompt: str, model: str = "gemini-2.0-flash") -> Dict[str, Any]:
        response = await self._gemini_model.generate_content_async(prompt)
        result = json.loads(response.text)
        tokens = response.usage_metadata.total_token_count if response.usage_metadata else 0
        return {
            "score": int(result["score"]),
            "reasoning": result.get("reasoning", ""),
            "model": model,
            "tokens": tokens,
        }

    # ------------------------------------------------------------------
    # Agreement metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _cohens_kappa_single(score_a: int, score_b: int, scale: int = 5) -> float:
        """
        Per-pair Cohen's Kappa approximation.
        For batch-level reliability, use calculate_batch_kappa() instead.
        Here we treat exact match as observed agreement (Po=1 or 0) and
        compute Pe assuming uniform distribution over the scale.
        """
        po = 1.0 if score_a == score_b else 0.0
        pe = 1.0 / scale
        if pe == 1.0:
            return 1.0
        return (po - pe) / (1.0 - pe)

    @staticmethod
    def calculate_batch_kappa(scores_a: List[int], scores_b: List[int], scale: int = 5) -> float:
        """
        Cohen's Kappa across a full batch of judge pairs.
        Use this after running the full benchmark for a reliable reliability estimate.
        """
        assert len(scores_a) == len(scores_b) and len(scores_a) > 0
        n = len(scores_a)
        po = sum(1 for a, b in zip(scores_a, scores_b) if a == b) / n

        from collections import Counter
        count_a = Counter(scores_a)
        count_b = Counter(scores_b)
        categories = range(1, scale + 1)
        pe = sum((count_a.get(k, 0) / n) * (count_b.get(k, 0) / n) for k in categories)

        if pe == 1.0:
            return 1.0
        return (po - pe) / (1.0 - pe)

    # ------------------------------------------------------------------
    # Conflict resolution
    # ------------------------------------------------------------------

    async def _resolve_conflict(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        score_a: int,
        score_b: int,
    ) -> Tuple[float, str]:
        """
        Khi 2 judge lệch nhau > 1 điểm, gọi tiebreaker (gpt-4o-mini) để phân xử.
        Trả về (final_score, method).
        """
        prompt = _SCORING_PROMPT.format(
            question=question, answer=answer, ground_truth=ground_truth
        )
        try:
            tiebreaker = await self._call_gpt(prompt, model="gpt-4o-mini")
            scores = sorted([score_a, score_b, tiebreaker["score"]])
            return float(scores[1]), "tiebreaker_median"
        except Exception:
            return (score_a * 0.6 + score_b * 0.4), "weighted_average"

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------

    async def evaluate_multi_judge(
        self, question: str, answer: str, ground_truth: str
    ) -> Dict[str, Any]:
        """
        Gọi GPT-4o và Gemini song song, tính Agreement Rate + Cohen's Kappa,
        xử lý xung đột tự động nếu lệch > 1 điểm.
        """
        prompt = _SCORING_PROMPT.format(
            question=question, answer=answer, ground_truth=ground_truth
        )

        result_gpt, result_gemini = await asyncio.gather(
            self._call_gpt(prompt, model="gpt-4o"),
            self._call_gemini(prompt),
        )

        score_gpt = result_gpt["score"]
        score_gemini = result_gemini["score"]
        diff = abs(score_gpt - score_gemini)

        exact_agreement = diff == 0
        near_agreement = diff <= 1
        agreement_rate = 1.0 if exact_agreement else (0.5 if near_agreement else 0.0)
        kappa = self._cohens_kappa_single(score_gpt, score_gemini)

        conflict_resolved = False
        resolution_method = "average"
        if diff > 1:
            final_score, resolution_method = await self._resolve_conflict(
                question, answer, ground_truth, score_gpt, score_gemini
            )
            conflict_resolved = True
        else:
            final_score = (score_gpt + score_gemini) / 2.0

        total_tokens = result_gpt["tokens"] + result_gemini["tokens"]

        return {
            "final_score": final_score,
            "agreement_rate": agreement_rate,
            "cohens_kappa": kappa,
            "individual_scores": {
                "gpt-4o": score_gpt,
                "gemini-2.0-flash": score_gemini,
            },
            "reasoning": {
                "gpt-4o": result_gpt["reasoning"],
                "gemini-2.0-flash": result_gemini["reasoning"],
            },
            "conflict_resolved": conflict_resolved,
            "resolution_method": resolution_method,
            "cost_tokens": total_tokens,
        }

    # ------------------------------------------------------------------
    # Position bias detection
    # ------------------------------------------------------------------

    async def check_position_bias(
        self, question: str, response_a: str, response_b: str, ground_truth: str = ""
    ) -> Dict[str, Any]:
        """
        Kiểm tra Position Bias: đánh giá cặp (A, B) và (B, A).
        Nếu judge luôn chọn câu ở vị trí đầu tiên → có position bias.
        """
        prompt_ab = _PAIRWISE_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            label_a="A", answer_a=response_a,
            label_b="B", answer_b=response_b,
        )
        prompt_ba = _PAIRWISE_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            label_a="B", answer_a=response_b,
            label_b="A", answer_b=response_a,
        )

        (res_ab_gpt, res_ba_gpt), (res_ab_gemini, res_ba_gemini) = await asyncio.gather(
            asyncio.gather(
                self._call_gpt_pairwise(prompt_ab),
                self._call_gpt_pairwise(prompt_ba),
            ),
            asyncio.gather(
                self._call_gemini_pairwise(prompt_ab),
                self._call_gemini_pairwise(prompt_ba),
            ),
        )

        def _is_biased(forward: str, backward: str) -> bool:
            return (forward == "A" and backward == "B") or (forward == "B" and backward == "A")

        biased_gpt = _is_biased(res_ab_gpt, res_ba_gpt)
        biased_gemini = _is_biased(res_ab_gemini, res_ba_gemini)

        return {
            "gpt_position_biased": biased_gpt,
            "gemini_position_biased": biased_gemini,
            "overall_biased": biased_gpt or biased_gemini,
            "detail": {
                "gpt": {"ab": res_ab_gpt, "ba": res_ba_gpt},
                "gemini": {"ab": res_ab_gemini, "ba": res_ba_gemini},
            },
        }

    async def _call_gpt_pairwise(self, prompt: str) -> str:
        response = await self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        return result.get("winner", "tie")

    async def _call_gemini_pairwise(self, prompt: str) -> str:
        response = await self._gemini_model.generate_content_async(prompt)
        result = json.loads(response.text)
        return result.get("winner", "tie")


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    async def _test():
        judge = LLMJudge()

        print("=== Test evaluate_multi_judge ===")
        result = await judge.evaluate_multi_judge(
            question="Làm thế nào để đổi mật khẩu tài khoản?",
            answer="Vào Cài đặt, chọn Bảo mật, rồi nhấn Đổi mật khẩu và nhập mật khẩu mới.",
            ground_truth="Truy cập Cài đặt > Bảo mật > Đổi mật khẩu. Nhập mật khẩu cũ và mật khẩu mới, xác nhận.",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))

        print("\n=== Test position bias ===")
        bias = await judge.check_position_bias(
            question="Cách tốt nhất để học lập trình là gì?",
            response_a="Học qua dự án thực tế và luyện tập hàng ngày.",
            response_b="Đọc sách giáo khoa từ đầu đến cuối trước khi code.",
            ground_truth="Kết hợp lý thuyết và thực hành, ưu tiên dự án thực tế.",
        )
        print(json.dumps(bias, ensure_ascii=False, indent=2))

        print("\n=== Test batch Cohen's Kappa ===")
        scores_a = [4, 3, 5, 2, 4, 3, 5, 4, 3, 2]
        scores_b = [4, 4, 5, 2, 3, 3, 4, 4, 3, 3]
        kappa = LLMJudge.calculate_batch_kappa(scores_a, scores_b)
        print(f"Batch Cohen's Kappa: {kappa:.4f}")

    asyncio.run(_test())
