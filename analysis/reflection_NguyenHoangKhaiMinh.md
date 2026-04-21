# Reflection Report - Nguyen Hoang Khai Minh

## 1) Vai trò và phạm vi đóng góp

Trong Lab Day 14, em đảm nhận vai trò **Retrieval Metrics Engineer**. Em chịu trách nhiệm chính cho toàn bộ các file trong thư mục `agent/` (bao gồm pipeline ingest, nhúng, truy xuất, rerank, sinh câu trả lời) và phát triển, hoàn thiện file `engine/retrieval_eval.py` để đánh giá chất lượng truy xuất (Retrieval).

Cụ thể:
- Đảm bảo pipeline ingest tài liệu, nhúng, lưu trữ vector, truy xuất và rerank hoạt động trơn tru, có thể debug từng bước chunking/indexing/top-k.
- Sửa lỗi và chuẩn hóa luồng truyền chunk_id/source qua các module để phục vụ đánh giá.
- Phát triển và hoàn thiện các hàm tính toán **Hit Rate** và **MRR** trong `retrieval_eval.py`, hỗ trợ cả hai chế độ đánh giá theo ID và theo context overlap.
- Viết hàm `evaluate_batch` để benchmark tự động trên 50+ test cases.

## 2) Engineering Contribution

- Refactor và fix lỗi import, truyền metadata (chunk_id, source, section) xuyên suốt pipeline.
- Đảm bảo khi ingest, mỗi chunk đều có chunk_id duy nhất, giúp debug và đánh giá retrieval chính xác.
- Sửa lỗi không in được chunk id khi chạy agent, đảm bảo context trả về luôn có chunk_id.
- Phát triển các hàm đánh giá retrieval: `calculate_hit_rate`, `calculate_mrr`, `calculate_context_hit_rate`, `calculate_context_mrr`, `evaluate_batch`, `evaluate_case`.
- Hỗ trợ team debug các vấn đề về mapping ID, context overlap, và kiểm tra lại logic top-k retrieval.

## 3) Technical Depth

- Hiểu rõ mối liên hệ giữa **Retrieval Quality** (Hit Rate, MRR) và **Answer Quality**: Nếu retrieval không lấy đúng chunk, LLM sẽ dễ trả lời sai/hallucinate.
- Phân biệt rõ hai chế độ đánh giá: theo ID (ground-truth) và theo context overlap (fallback khi không có ID).
- Đảm bảo pipeline retrieval có thể debug từng bước, giúp team xác định lỗi do truy xuất hay do sinh câu trả lời.
- Đề xuất chuẩn hóa metadata xuyên suốt pipeline để phục vụ phân tích lỗi và failure clustering.

## 4) Problem Solving

- **Vấn đề:** Không truyền được chunk_id/source qua các module → **Giải pháp:** Sửa lại logic add/search trong VectorStore, refactor Generator để truyền đủ metadata.
- **Vấn đề:** Lỗi import khi chạy từ các entrypoint khác nhau → **Giải pháp:** Chuẩn hóa lại import tuyệt đối cho toàn bộ agent/llm/*.
- **Vấn đề:** Đánh giá retrieval không nhất quán khi thiếu ground-truth ID → **Giải pháp:** Bổ sung chế độ context overlap, validate lại mapping ID.
- **Vấn đề:** Khó debug top-k retrieval → **Giải pháp:** In rõ chunk_id, source, score cho từng context trả về.

## 5) Kết quả, bài học và hướng cải tiến

### Kết quả
- Toàn bộ pipeline agent hoạt động ổn định, có thể ingest, truy xuất, rerank và sinh câu trả lời.
- Đã benchmark thành công Hit Rate & MRR trên 50+ test cases, hỗ trợ team phân tích lỗi retrieval vs. generation.
- Đảm bảo mọi context trả về đều có chunk_id/source, phục vụ failure analysis và clustering.

### Bài học
- Retrieval tốt là điều kiện tiên quyết để LLM trả lời đúng; nếu retrieval sai thì mọi cải tiến LLM đều vô nghĩa.
- Chuẩn hóa metadata và debug xuyên suốt pipeline giúp tiết kiệm rất nhiều thời gian phân tích lỗi.
- Cần luôn kiểm tra lại mapping ID và logic top-k để tránh đánh giá sai lệch.

### Hướng cải tiến
- Tự động hóa thêm các script kiểm tra consistency giữa ground-truth ID và index thực tế.
- Đề xuất bổ sung thêm các chỉ số như Recall@k, Precision@k cho các bài benchmark lớn hơn.
- Hỗ trợ thêm visualization cho pipeline retrieval để debug trực quan hơn.
