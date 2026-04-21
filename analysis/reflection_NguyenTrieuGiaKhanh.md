# Reflection Report - Nguyen Trieu Gia Khanh

**Họ và tên:** Nguyễn Triệu Gia Khánh  
**Mã sinh viên:** 2A202600225

## 1) Vai trò và phạm vi đóng góp

Trong Lab Day 14, em đảm nhận vai trò **Benchmark & Evaluation Integration Engineer**. Em chịu trách nhiệm hoàn thiện pipeline benchmark tổng thể để hệ thống có thể chạy ổn định từ project root, theo dõi đầy đủ chi phí/hiệu năng, so sánh regression giữa các phiên bản agent, và tạo ra bộ report cuối cùng phục vụ nộp bài. 

Ngoài phần việc chính của mình, em có hỗ trợ rà soát và sửa một số lỗi tích hợp ở các module do các bạn khác trong nhóm phụ trách, đặc biệt tại các điểm nối giữa agent, retrieval, judge và benchmark engine để toàn bộ hệ thống chạy được end-to-end.

Cụ thể:
- Sửa benchmark pipeline để chạy được trực tiếp từ `main.py` ở project root mà không bị lỗi import.
- Tích hợp theo dõi **latency**, **token usage**, và **estimated cost** xuyên suốt các giai đoạn agent, evaluator và multi-judge.
- Sửa logic cộng dồn token/cost của judge để các lượt **tie-breaker** cũng được tính đúng vào tổng chi phí benchmark.
- Hoàn thiện flow benchmark `V1 vs V2`, bổ sung release gate và quyết định `APPROVE/BLOCK_RELEASE`.
- Chạy benchmark trên 50 test cases, sinh report JSON và cập nhật report nhóm `analysis/failure_analysis.md`.

## 2) Engineering Contribution

- Refactor benchmark flow trong `main.py` để benchmark có thể chạy ổn định từ project root.
- Bổ sung runtime validation trước khi chạy benchmark, giúp phát hiện sớm thiếu API key hoặc dữ liệu benchmark.
- Hoàn thiện `engine/runner.py` để ghi nhận:
  - latency breakdown theo từng giai đoạn,
  - token usage cho agent/judge,
  - tổng token mỗi case,
  - estimated cost mỗi case.
- Sửa `engine/llm_judge.py` để:
  - luôn có 2 judge chạy song song,
  - xử lý conflict bằng tie-breaker,
  - cộng cả token tie-breaker vào thống kê cuối.
- Hoàn thiện regression benchmarking với 2 cấu hình agent:
  - `Agent_V1_Base`
  - `Agent_V2_Optimized`
- Bổ sung progress logging theo từng case để benchmark dài vẫn theo dõi được tiến độ và dễ debug khi có lỗi runtime.
- Hỗ trợ sửa và chuẩn hóa một số phần tích hợp giữa các module do các thành viên khác phát triển, nhằm tránh lệch contract dữ liệu giữa agent output, retrieval metrics, judge accounting và benchmark reporting.
- Chạy lại benchmark thực tế, sinh mới:
  - `reports/summary.json`
  - `reports/benchmark_results.json`
  - `analysis/failure_analysis.md`
- Kiểm tra đầu ra bằng `check_lab.py` để đảm bảo bộ file nộp bài hợp lệ.

## 3) Technical Depth

- Hiểu và triển khai được mối liên hệ giữa **performance metrics** (latency, token, cost) và **quality metrics** (judge score, agreement rate, hit rate, MRR) trong một benchmark pipeline hoàn chỉnh.
- Nắm rõ cách đánh giá độ tin cậy của hệ thống multi-judge thông qua:
  - Agreement Rate
  - Cohen's Kappa
  - conflict resolution bằng tie-breaker
- Hiểu rõ trade-off giữa chất lượng và chi phí khi so sánh cấu hình benchmark của `V1` và `V2`.
- Xử lý được bài toán regression testing thực tế: không chỉ benchmark một agent đơn lẻ mà còn phải chứng minh phiên bản tối ưu có cải thiện so với baseline.
- Hiểu tầm quan trọng của report tự động: benchmark chỉ có giá trị khi số liệu được ghi lại đầy đủ, nhất quán và có thể dùng trực tiếp cho submission.

## 4) Problem Solving

- **Vấn đề:** Benchmark không chạy được từ project root do lỗi import giữa các module.  
  **Giải pháp:** Chuẩn hóa lại import để toàn bộ agent/engine hoạt động thống nhất khi gọi từ `main.py`.

- **Vấn đề:** Benchmark có score nhưng thiếu theo dõi hiệu năng và chi phí theo từng case.  
  **Giải pháp:** Bổ sung latency breakdown, token accounting và estimated cost vào `runner`.

- **Vấn đề:** Token/cost của judge chưa phản ánh đúng khi có conflict và tie-breaker.  
  **Giải pháp:** Sửa logic cộng dồn trong `llm_judge.py` để tính đầy đủ token của cả primary judge, secondary judge và tie-breaker.

- **Vấn đề:** Regression ban đầu chỉ đổi tên version nhưng chưa phải so sánh có ý nghĩa.  
  **Giải pháp:** Thiết lập hai profile benchmark khác nhau cho `V1` và `V2`, từ đó tạo delta thực và release gate tự động.

- **Vấn đề:** Benchmark chạy lâu và khó biết đang dừng ở đâu.  
  **Giải pháp:** Thêm progress log theo từng case, đồng thời điều chỉnh concurrency để benchmark chạy ổn định hơn.

- **Vấn đề:** Một số module riêng lẻ chạy được nhưng ghép lại thì phát sinh lỗi tích hợp giữa output của agent, retrieval metrics và benchmark engine.  
  **Giải pháp:** Rà soát lại contract dữ liệu giữa các phần do nhiều thành viên phụ trách và sửa các điểm lệch để pipeline chạy thông suốt từ benchmark đến report.

- **Vấn đề:** Report nhóm ban đầu còn mang tính placeholder, chưa phản ánh kết quả benchmark thật.  
  **Giải pháp:** Chạy benchmark lại hoàn chỉnh và ghi trực tiếp kết quả mới vào report nhóm và các file report JSON.

## 5) Kết quả, bài học và hướng cải tiến

### Kết quả
- Benchmark pipeline đã chạy ổn định trên 50 test cases và sinh đầy đủ report phục vụ nộp bài.
- Đã theo dõi được đầy đủ latency, token usage và estimated cost cho toàn bộ pipeline.
- Regression benchmark cho kết quả:
  - `V1 Score = 4.13`
  - `V2 Score = 4.39`
  - `Delta = +0.26`
  - `Release Gate = APPROVE`
- Report nhóm đã được cập nhật với số liệu benchmark mới, failure clustering và 5 Whys.
- `check_lab.py` pass hoàn toàn, xác nhận bài đã sẵn sàng để chấm điểm.

### Bài học
- Một benchmark tốt không chỉ đo chất lượng đầu ra mà còn phải đo được chi phí, độ trễ và độ ổn định khi chạy thực tế.
- Multi-judge chỉ có ý nghĩa khi accounting và conflict resolution được triển khai đầy đủ, nếu không số liệu chi phí sẽ bị sai lệch.
- Regression testing cần một baseline thật sự khác biệt với bản tối ưu; nếu không, delta tạo ra sẽ không có giá trị kỹ thuật.
- Report tự động giúp tiết kiệm rất nhiều thời gian ở giai đoạn chốt bài và tránh sai sót khi tổng hợp thủ công.

### Hướng cải tiến
- Tạo script benchmark có thể resume hoặc checkpoint khi chạy lâu để tránh phải chạy lại toàn bộ nếu có lỗi giữa chừng.
- Chuẩn hóa sâu hơn phần release gate bằng các ngưỡng chất lượng/cost được cấu hình ngoài file code.
- Bổ sung thêm dashboard hoặc visualization cho benchmark results để việc phân tích regression trực quan hơn.
- Tiếp tục tối ưu baseline và candidate agent để kết quả regression phản ánh rõ hơn tác động của từng cải tiến kỹ thuật.
