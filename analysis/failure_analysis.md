# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 1. Tổng quan Benchmark
- **Tổng số cases:** 50
- **Tỉ lệ Pass/Fail:** 43/7
- **Điểm RAGAS trung bình:**
  - Faithfulness: 0.5905
  - Relevancy: 0.7233
  - Hit Rate: 1.0000
  - MRR: 0.9900
- **Điểm LLM-Judge trung bình:** 4.39 / 5.0
- **Agreement Rate:** 95.00%
- **Batch Cohen's Kappa:** 0.7754
- **Avg latency / case:** 3.52s
- **Avg estimated cost / case:** $0.0043

## 2. Phân nhóm lỗi (Failure Clustering)
| Nhóm lỗi | Số lượng | Nguyên nhân dự kiến |
|----------|----------|---------------------|
| policy-check | 2 | Retrieval mode: context_overlap_fallback; needs tighter context selection or stronger refusal behavior. |
| fact-check | 1 | Retrieval mode: context_overlap_fallback; needs tighter context selection or stronger refusal behavior. |
| adversarial-prompt-injection | 1 | Retrieval mode: context_overlap_fallback; needs tighter context selection or stronger refusal behavior. |
| adversarial-goal-hijacking | 1 | Retrieval mode: context_overlap_fallback; needs tighter context selection or stronger refusal behavior. |
| conflicting-information-resolution | 1 | Retrieval mode: context_overlap_fallback; needs tighter context selection or stronger refusal behavior. |

## 3. Phân tích 5 Whys (Chọn 3 case tệ nhất)

### Case #1: Phí giữ học bổng có được bù trừ vào khoản nào?
1. **Symptom:** Judge score 1.50/5, faithfulness 0.00, retrieval hit rate 1.00.
2. **Why 1:** Câu hỏi thuộc nhóm `policy-check` với độ khó `easy`, nên rất nhạy với context sai hoặc thiếu.
3. **Why 2:** Retrieval đang chạy ở chế độ `context_overlap_fallback`; nếu không hit ở rank đầu, answer quality giảm rõ rệt.
4. **Why 3:** Agent trả về 3 context, nên nhiễu retrieval hoặc chunk quá rộng vẫn có thể kéo tụt precision.
5. **Why 4:** Judge agreement = 0.50; khi agreement chưa tuyệt đối, câu trả lời thường đúng một phần nhưng chưa bám sát policy wording.
6. **Root Cause:** Cần tiếp tục tối ưu mapping từ câu hỏi khó sang chunk giàu tín hiệu hơn, đồng thời siết cách agent trả lời cho các case adversarial/ambiguous.
### Case #2: Phí giữ học bổng/hỗ trợ tài chính là bao nhiêu?
1. **Symptom:** Judge score 1.50/5, faithfulness 0.00, retrieval hit rate 1.00.
2. **Why 1:** Câu hỏi thuộc nhóm `fact-check` với độ khó `easy`, nên rất nhạy với context sai hoặc thiếu.
3. **Why 2:** Retrieval đang chạy ở chế độ `context_overlap_fallback`; nếu không hit ở rank đầu, answer quality giảm rõ rệt.
4. **Why 3:** Agent trả về 3 context, nên nhiễu retrieval hoặc chunk quá rộng vẫn có thể kéo tụt precision.
5. **Why 4:** Judge agreement = 0.50; khi agreement chưa tuyệt đối, câu trả lời thường đúng một phần nhưng chưa bám sát policy wording.
6. **Root Cause:** Cần tiếp tục tối ưu mapping từ câu hỏi khó sang chunk giàu tín hiệu hơn, đồng thời siết cách agent trả lời cho các case adversarial/ambiguous.
### Case #3: (Cross-section) Có thể nói mọi khoản phí đều không hoàn trả không?
1. **Symptom:** Judge score 2.00/5, faithfulness 0.00, retrieval hit rate 1.00.
2. **Why 1:** Câu hỏi thuộc nhóm `cross-section-conflict` với độ khó `hard`, nên rất nhạy với context sai hoặc thiếu.
3. **Why 2:** Retrieval đang chạy ở chế độ `context_overlap_fallback`; nếu không hit ở rank đầu, answer quality giảm rõ rệt.
4. **Why 3:** Agent trả về 3 context, nên nhiễu retrieval hoặc chunk quá rộng vẫn có thể kéo tụt precision.
5. **Why 4:** Judge agreement = 1.00; khi agreement chưa tuyệt đối, câu trả lời thường đúng một phần nhưng chưa bám sát policy wording.
6. **Root Cause:** Cần tiếp tục tối ưu mapping từ câu hỏi khó sang chunk giàu tín hiệu hơn, đồng thời siết cách agent trả lời cho các case adversarial/ambiguous.

## 4. Kế hoạch cải tiến (Action Plan)
- [x] Đồng bộ benchmark để chạy từ project root và ghi đầy đủ latency/token/cost vào report.
- [x] Bổ sung fallback retrieval evaluation bằng context overlap khi curated `ground_truth_ids` không cùng namespace với runtime chunk IDs.
- [x] Thiết lập regression gate với quyết định hiện tại: **APPROVE**.
- [ ] Chuẩn hóa lại hệ chunk ID trong vector store để khớp 1-1 với curated ground-truth IDs của dataset.
- [ ] Tạo thêm một baseline agent riêng biệt ở code-level để regression phản ánh chính xác từng thay đổi kiến trúc hơn nữa.