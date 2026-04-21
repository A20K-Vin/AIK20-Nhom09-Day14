# Reflection Report - Nguyen Thi Dieu Linh

**Họ và tên:** Nguyen Thi Dieu Linh   
**Mã sinh viên:** 2A202600209

## 1) Vai trò và phạm vi tham gia

Trong Lab Day 14 (AI Evaluation Factory), tôi tham gia chủ yếu ở mức quan sát và hỗ trợ kỹ thuật cơ bản: chạy thử các script, đọc cấu trúc code và theo dõi quá trình nhóm triển khai pipeline evaluation.

Cụ thể, tôi tập trung vào việc hiểu luồng tổng thể của hệ thống, từ bước tạo Golden Dataset đến khi sinh ra báo cáo benchmark. Tôi chưa đảm nhận phần implementation như xây dựng eval engine hay multi-judge, nhưng đã chủ động đọc các module liên quan để nắm được cách hệ thống vận hành end-to-end.

## 2) Những gì học được
Hiểu rõ pipeline evaluation hoàn chỉnh của một hệ thống RAG: từ Synthetic Data Generation → Retrieval Evaluation → Multi-Judge → Benchmark → Failure Analysis. Điều này giúp tôi nhận ra evaluation là thành phần trung tâm để cải tiến hệ thống, không phải bước phụ.   
Nắm được vai trò của retrieval metrics như Hit Rate và MRR. Trong đó, MRR không chỉ đo việc retrieve đúng hay không, mà còn phản ánh vị trí của kết quả đúng trong ranking, điều này ảnh hưởng trực tiếp đến khả năng model sử dụng đúng thông tin trong bước generation.   
Hiểu được khái niệm Position Bias: các kết quả ở vị trí đầu danh sách có xu hướng được ưu tiên sử dụng hơn, vì vậy một hệ thống có MRR cao thường dẫn đến chất lượng câu trả lời tốt hơn ngay cả khi Hit Rate tương đương.   
Hiểu được vai trò của multi-judge consensus và sự cần thiết của việc đo độ đồng thuận giữa các judge. Cụ thể, các chỉ số như Cohen’s Kappa có thể được sử dụng để đánh giá mức độ nhất quán giữa các model judge, từ đó phản ánh độ tin cậy của hệ thống đánh giá.   
Nhận ra trade-off giữa chi phí và chất lượng: việc sử dụng nhiều judge model hoặc model mạnh hơn giúp tăng độ chính xác đánh giá, nhưng làm tăng chi phí và latency. Vì vậy, cần thiết kế hệ thống cân bằng giữa độ tin cậy và hiệu quả tài nguyên.   
Hiểu tầm quan trọng của golden dataset có ground truth rõ ràng: nếu dữ liệu không chuẩn hóa, toàn bộ pipeline evaluation phía sau sẽ không còn ý nghĩa.   
Biết cách đọc báo cáo benchmark để xác định bottleneck: phân biệt được lỗi đến từ retrieval, ranking hay generation thay vì chỉ nhìn vào output cuối.   

## 3) Vấn đề gặp phải và cách nhìn nhận

* Tôi gặp khó khăn khi đọc các module liên quan đến judge (đặc biệt là parsing JSON từ LLM), do thiếu kinh nghiệm với prompt design và xử lý output không ổn định.

* Khi theo dõi pipeline, tôi nhận ra một vấn đề quan trọng: hệ thống phụ thuộc nhiều vào assumption rằng LLM trả về đúng format, trong khi thực tế điều này không luôn đảm bảo.

* Từ đó, tôi hiểu rằng việc thiết kế prompt và cơ chế fallback (retry, validation) là yếu tố bắt buộc trong hệ thống production, không chỉ là chi tiết phụ.

## 4) Tự đánh giá và kế hoạch cải thiện

Tôi nhận thấy hạn chế lớn nhất của mình là chưa tham gia đủ sâu vào phần implementation, dẫn đến đóng góp thực tế còn ít. Tuy nhiên, tôi đã tận dụng thời gian để xây dựng hiểu biết nền tảng về hệ thống evaluation, thay vì chỉ theo dõi một cách thụ động.

Trong các bước tiếp theo, tôi sẽ:

1. Chạy lại toàn bộ pipeline (`synthetic_gen.py`, `main.py`) để hiểu rõ luồng thực thi.
2. Đọc kỹ `golden_set.jsonl` và tự tạo thêm một số test cases để nắm format dữ liệu.
3. Thực hiện một PR nhỏ (documentation hoặc validation script) để bắt đầu đóng góp trực tiếp.
4. Thực hành viết prompt đơn giản cho judge và kiểm tra khả năng parse JSON ổn định.
5. Tham gia chạy benchmark và so sánh kết quả giữa các phiên bản để hiểu rõ regression testing.
