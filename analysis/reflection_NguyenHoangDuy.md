# Reflection Report - Nguyen Hoang Duy

**Họ và tên:** Nguyễn Hoàng Duy
**Mã sinh viên:** 2A202600158

## 1) Vai trò và phạm vi đóng góp

Trong Lab Day 14, em đảm nhận vai trò **Multi-Judge Consensus Engineer**. Em chịu trách nhiệm thiết kế và triển khai toàn bộ module `engine/llm_judge.py` — thành phần cốt lõi đảm bảo kết quả đánh giá của hệ thống là khách quan, có thể đo lường độ tin cậy, và xử lý được các trường hợp bất đồng giữa các judge.

Cụ thể:
- Thay thế hoàn toàn phần placeholder bằng implementation thực gọi API.
- Thiết kế prompt rubric chi tiết cho cả scoring đơn lẻ và so sánh pairwise.
- Triển khai hai judge chạy song song bằng `asyncio.gather`.
- Xây dựng hệ thống đo Agreement Rate và Cohen's Kappa ở cả mức per-pair lẫn batch.
- Triển khai cơ chế tự động giải quyết xung đột khi hai judge lệch nhau trên 1 điểm.
- Triển khai kiểm tra Position Bias bằng phương pháp hoán đổi thứ tự (AB/BA swap).
- Theo dõi token usage đầy đủ bao gồm cả tiebreaker, phục vụ báo cáo chi phí.
- Kết nối `LLMJudge` vào `main.py` thay cho class placeholder `MultiModelJudge`.

## 2) Engineering Contribution

- Thiết kế `_SCORING_PROMPT` và `_PAIRWISE_PROMPT` với rubric rõ ràng 5 mức, yêu cầu trả về JSON thuần, giảm thiểu lỗi parse.
- Triển khai `_call_gpt()` với `response_format={"type": "json_object"}` để đảm bảo GPT-4o trả về đúng định dạng.
- Triển khai `_call_secondary_judge()` dùng GPT-4o-mini làm judge thứ hai — đủ tiêu chí 2 judge độc lập theo yêu cầu rubric, đồng thời tiết kiệm chi phí.
- Triển khai `_cohens_kappa_single()` cho đánh giá per-pair và `calculate_batch_kappa()` (static method) để tính Cohen's Kappa toàn bộ benchmark sau khi chạy xong.
- Cơ chế conflict resolution trong `_resolve_conflict()`: gọi tiebreaker GPT-4o-mini, lấy **median of 3** scores; fallback về weighted average nếu tiebreaker lỗi.
- `check_position_bias()`: chạy song song 4 lượt gọi API (AB và BA cho cả 2 judge), phát hiện judge có xu hướng thiên vị vị trí xuất hiện của câu trả lời.
- Token tracking đầy đủ: `token_usage` trong output ghi nhận riêng token của primary judge, secondary judge và tiebreaker, tổng hợp vào `cost_tokens`.

## 3) Technical Depth

- **Agreement Rate vs Cohen's Kappa:** Agreement Rate đơn giản (tỷ lệ đồng điểm) có thể bị thổi phồng khi cả hai judge ngẫu nhiên cho cùng điểm. Cohen's Kappa loại trừ phần đồng thuận do ngẫu nhiên (expected agreement Pe), nên là chỉ số đáng tin cậy hơn để đánh giá độ nhất quán giữa hai judge.
- **Position Bias:** LLM judge có xu hướng ưu tiên câu trả lời xuất hiện ở vị trí đầu tiên trong prompt, không phải vì nó tốt hơn mà vì attention pattern của model. Phương pháp AB/BA swap phát hiện bias này bằng cách so sánh kết quả khi đổi thứ tự hai câu trả lời.
- **Median of 3 cho conflict resolution:** Khi hai judge lệch nhau nhiều, trung bình cộng không hợp lý vì kéo điểm về giữa một cách cơ học. Tiebreaker thứ ba + lấy median loại bỏ được outlier, cho kết quả phản ánh đúng hơn quan điểm đa số.
- **Trade-off chi phí:** GPT-4o-mini làm secondary judge thay vì Gemini Pro giúp giảm chi phí đáng kể (~10x rẻ hơn GPT-4o) trong khi vẫn đủ năng lực đánh giá cho bài toán scoring 1-5. Token tiebreaker được track riêng để có thể phân tích chi phí theo từng kịch bản conflict.

## 4) Problem Solving

- **Vấn đề:** Model có thể trả về JSON bị wrap trong markdown code block hoặc text thừa, gây lỗi parse.
  **Giải pháp:** Dùng `response_format={"type": "json_object"}` cho OpenAI để buộc output là JSON thuần, thiết kế prompt nêu rõ "không thêm text ngoài JSON".

- **Vấn đề:** Ban đầu dùng Gemini làm secondary judge nhưng `google-generativeai` SDK có behavior không ổn định với JSON mode ở một số phiên bản.
  **Giải pháp:** Chuyển secondary judge sang GPT-4o-mini (cùng OpenAI SDK) để đảm bảo tính nhất quán. Kiến trúc `_call_secondary_judge()` tách biệt hoàn toàn với primary, dễ swap model sau này.

- **Vấn đề:** Token của tiebreaker không được tính vào tổng chi phí trong phiên bản đầu, làm lệch báo cáo cost.
  **Giải pháp:** `_resolve_conflict()` trả về tuple `(score, method, extra_tokens)`, tích hợp `tiebreaker_tokens` vào `total_tokens` và `token_usage` của output.

- **Vấn đề:** Position bias test cần chạy 4 lượt API call nhưng nếu chạy tuần tự thì chậm.
  **Giải pháp:** Dùng `asyncio.gather` lồng nhau để chạy song song cả 4 lượt, tổng thời gian xấp xỉ 1 lượt call đơn.

## 5) Kết quả, bài học và hướng cải tiến

### Kết quả
- Module `engine/llm_judge.py` hoàn chỉnh, tích hợp thành công vào benchmark pipeline.
- Hai judge (GPT-4o + GPT-4o-mini) chạy song song, output đầy đủ agreement rate, Cohen's Kappa, reasoning của từng judge và token usage chi tiết.
- Cơ chế conflict resolution hoạt động tự động, không cần can thiệp thủ công.
- Position bias detection sẵn sàng để gọi trong failure analysis.

### Bài học
- Một judge đơn lẻ không đủ tin cậy cho hệ thống production; sự bất đồng giữa các judge chứa thông tin quan trọng về độ khó và tính mơ hồ của từng test case.
- Cohen's Kappa nên được tính ở cấp batch chứ không phải per-pair; per-pair chỉ có ý nghĩa khi dùng để flag những case có bất đồng cao để review thủ công.
- Thiết kế prompt rubric rõ ràng với ví dụ điểm cụ thể giảm đáng kể variance giữa các lượt chấm của cùng một model.

### Hướng cải tiến
- Bổ sung third-party judge (ví dụ Gemini) khi SDK ổn định hơn để tăng tính độc lập giữa các judge.
- Tính Cohen's Kappa ở cấp batch và đưa vào `reports/summary.json` như một chỉ số reliability của toàn bộ đợt benchmark.
- Thêm confidence interval cho `final_score` dựa trên variance giữa các judge, giúp release gate đưa ra quyết định có cơ sở thống kê hơn.
