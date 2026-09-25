# Individual contribution report

## Thông tin

- Họ và tên: Đào Duy Hiếu
- Mã học viên: 2A202602651
- Vai trò: Full-stack Engineer
- Nhóm: Nhóm PatchLens (K4-L3B)
- Repository/branch: `ngaiTu29s1/K4-L3B-RAG-Pipeline` / `main`

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| **Golden Dataset (Task 11)** | Xây dựng bộ câu hỏi và đáp án chuẩn hóa gồm 20 case, bao phủ thông tin champion, item, summoner spell, Patch Notes 26.15–26.19 và câu hỏi ngoài domain để kiểm tra safe refusal. | `group_project/evaluation/golden_dataset.json` | Done |
| **Evaluation Runner (Task 12)** | Thiết kế và thực hiện đánh giá A/B giữa Dense-only và Hybrid + RRF trên cùng golden dataset, sử dụng 4 metrics: Faithfulness, Answer Relevance, Context Recall và Context Precision. | `group_project/evaluation/eval_results_dense.json`, `group_project/evaluation/eval_results_hybrid.json`, `group_project/evaluation/eval_summary.json` | Done |

## Quyết định kỹ thuật quan trọng

1. **Quyết định: Dùng golden dataset có cả câu hỏi in-domain và out-of-domain**  
   **Lý do/evidence:** Bộ dữ liệu cần kiểm tra đồng thời khả năng truy xuất thông tin Liên Minh Huyền Thoại và cơ chế safe refusal. 20 case được phân bổ trên nhiều nhóm dữ liệu: champion, item, summoner spell, các bản Patch Notes 26.15–26.19 và câu hỏi ngoài phạm vi. Cách này giúp đánh giá không chỉ câu trả lời đúng mà còn khả năng từ chối khi không có evidence phù hợp.

2. **Quyết định: So sánh hai cấu hình trên cùng dữ liệu và cùng generator**  
   **Lý do/evidence:** Config A sử dụng Dense-only; Config B kết hợp Dense Search, BM25 và RRF với `k=60`. Hai cấu hình dùng chung golden dataset 20 case, `top_k=5`, generator `gemini-3.5-flash-lite` và cơ chế citation validation, vì vậy chênh lệch phản ánh chủ yếu ảnh hưởng của chiến lược retrieval.

3. **Quyết định: Lưu kết quả chi tiết và summary riêng**  
   **Lý do/evidence:** Hai file kết quả theo cấu hình giúp kiểm tra từng case và phân tích lỗi, còn `eval_summary.json` cung cấp số liệu tổng hợp để đối chiếu nhanh giữa Dense-only và Hybrid + RRF. Cấu trúc này hỗ trợ tái lập evaluation và viết báo cáo nhóm.

## Kiểm thử và kết quả

- **Phạm vi kiểm thử:**
  - Golden dataset: **20 cases**, vượt yêu cầu tối thiểu 15 câu hỏi.
  - 4 metrics: Faithfulness, Answer Relevance, Context Recall và Context Precision.
  - A/B evaluation: Dense-only so với Hybrid + RRF trên cùng cấu hình còn lại.
  - Kiểm tra toàn bộ repository bằng `pytest -q`: **26/26 passed**.
- **Kết quả A/B:**
  - Dense-only: Faithfulness `0.8921`, Answer Relevance `0.6471`, Context Recall `1.0000`, Context Precision `0.8703`, trung bình `0.8524`.
  - Hybrid + RRF: Faithfulness `0.9055`, Answer Relevance `0.6941`, Context Recall `0.9833`, Context Precision `0.8513`, trung bình `0.8585`.
  - Hybrid + RRF tăng điểm trung bình `+0.0061`, Faithfulness `+0.0134` và Answer Relevance `+0.0470` so với Dense-only.
- **Kết luận:** Hybrid + RRF là cấu hình tốt hơn cho các truy vấn có token phiên bản patch và tên kỹ năng cụ thể; BM25 giúp giảm nhầm lẫn giữa các bản patch có nội dung ngữ nghĩa tương tự.

## Điều còn hạn chế

- **Một hạn chế cụ thể của phần tôi làm:** Golden dataset mới có 20 case, nên chưa bao phủ toàn bộ biến thể diễn đạt tiếng Việt, truy vấn follow-up và các trường hợp biên của từng loại tài liệu.
- **Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện:** Mở rộng dataset theo từng nhóm intent, bổ sung nhiều paraphrase Việt–Anh và thêm các test case kiểm tra temporal consistency giữa các phiên bản patch.
- **Lưu ý tái lập:** Evaluation phụ thuộc vào embedding/generator API và quota mạng; cần bảo đảm `.env` được cấu hình hợp lệ nhưng không commit API key vào repository.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 2026-09-25
- Tên thành viên: Đào Duy Hiếu
