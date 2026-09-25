# Danh sách thành viên nhóm — PatchLens RAG Pipeline

## 1. Thông tin chung dự án

- **Tên dự án:** PatchLens — RAG Pipeline tra cứu thông tin tướng, trang bị, phép bổ trợ và bản cập nhật Liên Minh Huyền Thoại
- **Khóa học:** K4-L3B
- **Repository:** `ngaiTu29s1/K4-L3B-RAG-Pipeline`
- **Nhánh chính:** `main`

---

## 2. Bảng phân công thành viên

| STT | Họ và tên | Mã học viên | Vai trò | Nhánh phụ trách | Phần việc đảm nhiệm chính |
| :---: | :--- | :---: | :--- | :---: | :--- |
| 1 | **Trần Tuấn Tú** | `2A202602840` | Full-stack RAG Engineer / Trưởng nhóm | `main` | Chịu trách nhiệm toàn bộ quy trình Pipeline end-to-end từ Task 1 đến Task 10 |
| 2 | **Đào Duy Hiếu** | `2A202602651` | Full-stack Engineer | `main` | Phụ trách Golden Dataset và Evaluation Runner cho Tasks 11–12 |

---

## 3. Chi tiết phần việc và đóng góp

### **Trần Tuấn Tú**
- **Mã học viên:** `2A202602840`
- **Nhánh thực hiện:** `main`
- **Các module trực tiếp triển khai:**
  1. **Thu thập & Chuẩn hóa dữ liệu (Tasks 1–3):** Thu thập Data Dragon (champions, items, summoner spells) và crawl 5 bản Patch Notes (26.15–26.19); chuẩn hóa thành 523 file Markdown sạch tại `data/standardized/`.
  2. **Chunking & Vector Indexing (Task 4):** Xây dựng bộ chia chunk đệ quy (500 ký tự, overlap 50), embedding đa chiều với `gemini-embedding-001`, lưu trữ 1.899 chunks vào ChromaDB persistent storage.
  3. **Hybrid Retrieval & Reranking (Tasks 5–7):** Phát triển Dense Semantic Search, BM25 Lexical Search và thuật toán Reciprocal Rank Fusion (RRF $k=60$) giúp kết hợp ưu điểm của cả hai phương pháp.
  4. **Fallback & Pipeline hoàn chỉnh (Tasks 8–9):** Tích hợp PageIndex vectorless search và hiệu chuẩn ngưỡng cosine fallback `0.66` (in-domain $\ge 0.7172$, out-of-domain $\le 0.6044$).
  5. **Generation có Citation (Task 10):** Tích hợp LLM `gemini-3.5-flash-lite`, xử lý Lost-in-the-middle reordering, ràng buộc trích dẫn chunk ID chính xác và Safe Refusal khi ngoài domain.
  6. **Golden Dataset (Task 11):** Xây dựng bộ câu hỏi chuẩn 20 case tại `group_project/evaluation/golden_dataset.json` bao phủ toàn bộ các chủ đề.
  7. **Evaluation Runner (Task 12):** Lập trình runner tự động (`src/task12_evaluation.py`) đo 4 metrics (Faithfulness, Answer Relevance, Context Recall, Context Precision) so sánh nhánh A/B.
  8. **Evaluation Report (Task 13):** Hoàn thành báo cáo đánh giá chuyên sâu `RESULT.md`, phân tích nguyên nhân lỗi metadata chunk-0 và đề xuất giải pháp.
  9. **Chatbot Streamlit UI & Final QA (Task 14):** Nâng cấp giao diện người dùng `app.py` với tính năng phân tách nguồn trích dẫn trực tiếp và ngữ cảnh nền; đưa toàn bộ 26/26 tests về trạng thái xanh.

### **Đào Duy Hiếu**
- **Mã học viên:** `2A202602651`
- **Vai trò:** Full-stack Engineer
- **Nhánh thực hiện:** `main`
- **Các module trực tiếp triển khai:**
  1. **Golden Dataset (Task 11):** Xây dựng bộ 20 câu hỏi và đáp án chuẩn hóa bao phủ champion, item, summoner spell, các bản Patch Notes 26.15–26.19 và câu hỏi ngoài domain tại `group_project/evaluation/golden_dataset.json`.
  2. **Evaluation Runner (Task 12):** Thực hiện đánh giá A/B giữa Dense-only và Hybrid + RRF bằng 4 metrics: Faithfulness, Answer Relevance, Context Recall và Context Precision; lưu kết quả tại `group_project/evaluation/eval_results_dense.json`, `group_project/evaluation/eval_results_hybrid.json` và `group_project/evaluation/eval_summary.json`.
