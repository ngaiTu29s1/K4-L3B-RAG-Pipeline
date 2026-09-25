# Individual contribution report

## Thông tin

- Họ và tên: Trần Tuấn Tú
- Mã học viên: 2A202602840
- Nhóm: Nhóm PatchLens (K4-L3B)
- Repository/branch: `ngaiTu29s1/K4-L3B-RAG-Pipeline` / `main`

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| **Data Collection & Standardization** | Thu thập Data Dragon (tướng, trang bị, phép bổ trợ) & crawl 5 bản Patch Notes (26.15–26.19); chuẩn hóa 523 file Markdown. | `data/landing/`, `data/standardized/` | Done |
| **Chunking & Vector Indexing** | Thiết kế recursive chunking (500/50), embedding Gemini 768-dim, index 1.899 chunks vào ChromaDB persistent storage. | `src/task4_chunking_indexing.py` | Done |
| **Hybrid Retrieval & Fallback** | Xây dựng Dense Search, BM25 Lexical Search, Reciprocal Rank Fusion (RRF $k=60$) và PageIndex fallback với ngưỡng 0.66. | `src/task5_semantic_search.py`, `src/task6_lexical_search.py`, `src/task7_reranking.py`, `src/task9_retrieval_pipeline.py` | Done |
| **Generation có Citation** | Pipeline sinh câu trả lời với `gemini-3.5-flash-lite`, Lost-in-the-middle reordering, bắt buộc trích dẫn chunk ID và safe refusal out-of-domain. | `src/task10_generation.py` | Done |
| **Golden Dataset (Task 11)** | Xây dựng bộ golden dataset 20 câu hỏi chuẩn hóa bao phủ champion, item, summoner spell, các bản patch và câu ngoài domain. | `group_project/evaluation/golden_dataset.json` | Done |
| **Evaluation Runner (Task 12)** | Xây dựng pipeline đánh giá A/B tự động giữa Dense-only và Hybrid + RRF qua 4 metrics (Faithfulness, Relevance, Recall, Precision). | `src/task12_evaluation.py`, `eval_results_*.json` | Done |
| **Evaluation Report (Task 13)** | Phân tích thực nghiệm A/B, đánh giá worst performers (lỗi metadata `chunk-0` lấn payload `chunk-1`) và đề xuất cải tiến. | `group_project/evaluation/RESULT.md`, `reports/RESULT.md` | Done |
| **Chatbot UI & QA (Task 14)** | Nâng cấp giao diện Streamlit, phân tách nguồn trích dẫn trực tiếp và ngữ cảnh ứng viên, đưa toàn bộ suite về 26/26 PASS. | `app.py`, `tests/` | Done |

## Quyết định kỹ thuật quan trọng

1. **Quyết định: Kết hợp Hybrid Retrieval (Dense Embeddings + BM25) bằng Reciprocal Rank Fusion (RRF $k=60$)**  
   **Lý do/evidence:** Dense search thuần túy dễ bị nhầm lẫn giữa các bản patch khác nhau của cùng một vị tướng (ví dụ truy vấn Volibear Patch 26.19 nhưng trả về chunk của Patch 26.15 vì cùng ngữ nghĩa). BM25 giúp bắt chính xác các token số phiên bản (`26.19`, `26.15`) và tên kỹ năng cụ thể, nâng Answer Relevance từ `0.6471` lên `0.6941` và Faithfulness lên `0.9055`.  
   **Trade-off:** Cần duy trì thêm cấu trúc index BM25 in-memory, tuy nhiên chi phí tính toán là không đáng kể (< 5ms cho 1.899 chunks) và không tốn thêm bất kỳ chi phí API embedding nào.

2. **Quyết định: Phân tách rõ ràng giữa "Nguồn trích dẫn trực tiếp" và "Ngữ cảnh ứng viên Top-K" trên giao diện người dùng**  
   **Lý do/evidence:** Bộ tìm kiếm Top-K luôn trả về các đoạn ngữ cảnh ứng viên tốt nhất (trong đó có thể có các chunk liên quan nhẹ hoặc tài liệu nền), trong khi LLM chỉ chọn lọc một số chunk chính xác nhất để trích dẫn. Nếu hiển thị toàn bộ 5 chunk ngang hàng sẽ gây hiểu nhầm rằng hệ thống dùng sai nguồn.  
   **Trade-off:** Cần bổ sung logic phân tích regex citation ID từ câu trả lời của mô hình để phân loại nguồn trước khi hiển thị trên giao diện Streamlit.

## Kiểm thử và kết quả

- **Test hoặc query đã dùng:**
  - Chạy toàn bộ test suite hợp đồng và nghiệm thu: `pytest -q` đạt **26/26 passed**.
  - Kiểm thử giao diện tự động: `pytest tests/test_app.py -q` pass.
  - Query mẫu kiểm chứng: `Patch 26.19 thay đổi nội tại The Relentless Storm của Volibear như thế nào?`, `Trang bị Vô Cực Kiếm cung cấp những chỉ số gì và giá mua là bao nhiêu?`, câu hỏi ngoài domain `Công thức nấu phở bò Hà Nội?`.
- **Kết quả trước/sau:**
  - Trước Task 11–13: Suite có 24 passed và 2 failed có chủ đích (`test_golden_dataset_has_15_grounded_cases` và `test_evaluation_report_is_completed`).
  - Sau Task 11–13: Toàn bộ 26/26 tests đều PASS 100%.
- **Lỗi đã phát hiện và cách xử lý:**
  - *Lỗi metadata chunk-0 lấn át payload chunk-1/chunk-2*: Khi chia nhỏ văn bản, chunk header chỉ chứa metadata chiếm mất 1 slot top-k; xử lý bằng cách phân tích trong báo cáo và khuyến nghị metadata injection.
  - *Lỗi quota RPM khi embed query hàng loạt*: Xử lý bằng cách batching 20 queries vào một request duy nhất trong `task12_evaluation.py` kèm cơ chế lưu cache.

## Điều còn hạn chế

- **Một hạn chế cụ thể của phần tôi làm:** Chunker cố định 500 ký tự đôi khi tách rời tiêu đề tài liệu khỏi bảng thông số kỹ năng nếu văn bản patch notes ngắn, khiến chunk payload độc lập thiếu từ khóa thực thể.
- **Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện:** Triển khai **Metadata Context Injection** (tự động gắn `[Entity: {Tên} | Patch: {Version}]` vào đầu mọi chunk con) và tích hợp **Cross-Encoder Reranker** (như `bge-reranker-base`) để tối ưu hóa thứ hạng trước khi đưa vào LLM.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 2026-09-25
- Tên thành viên: Trần Tuấn Tú
