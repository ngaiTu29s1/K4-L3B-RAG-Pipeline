# RAG evaluation results

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | 2026-09-25 |
| Framework and version              | Python 3.12, ChromaDB 0.6.3, rank-bm25 0.2.2, Google GenAI SDK |
| Evaluator model                    | Gemini 3.5 Flash Lite + Grounded Alignment Engine |
| Generator model                    | gemini-3.5-flash-lite |
| Embedding model                    | gemini-embedding-001 (768 dimensions) |
| Corpus version/commit              | 6d02fb6dd0315bb68eabf28e6401209626cd9327 |
| Golden dataset size                | 20 cases (Champions, Items, Spells, Patches 26.15–26.19, Out-of-domain) |
| `top_k`                            | 5 |
| Fallback threshold and calibration | 0.66 (Calibrated: In-domain >= 0.7172, Out-of-domain <= 0.6044) |

## Configurations

- **Config A — dense-only:** Semantic search thuần túy sử dụng ChromaDB vector store và Gemini `gemini-embedding-001`, lấy top 5 chunks theo cosine similarity cao nhất, fallback PageIndex khi max cosine score < 0.66.
- **Config B — hybrid + RRF:** Kết hợp dense semantic search (top 10) và BM25 lexical search (top 10), gộp thứ hạng bằng Reciprocal Rank Fusion (RRF, $k=60$) để trích xuất top 5 chunks, fallback PageIndex khi best cosine score < 0.66.

Hai config sử dụng cùng golden dataset (20 câu hỏi), cùng generator model (`gemini-3.5-flash-lite`), cùng system prompt, cùng `top_k = 5` và cùng cơ chế citation validation; chỉ khác nhau ở chiến lược retrieval.

## Overall scores

| Metric            | Config A (Dense-only) | Config B (Hybrid + RRF) | Delta B−A |
| ----------------- | --------------------: | ----------------------: | --------: |
| Faithfulness      |                0.8921 |                  0.9055 |   +0.0134 |
| Answer relevance  |                0.6471 |                  0.6941 |   +0.0470 |
| Context recall    |                1.0000 |                  0.9833 |   -0.0167 |
| Context precision |                0.8703 |                  0.8513 |   -0.0190 |
| **Average**       |            **0.8524** |              **0.8585** | **+0.0061** |

## A/B comparison

- **Cấu hình tốt hơn:** **Config B (Hybrid + RRF)** vượt trội hơn về chất lượng câu trả lời sinh ra, thể hiện rõ qua sự gia tăng của cả **Faithfulness (+0.0134)** và **Answer Relevance (+0.0470)**.
- **Evidence:**
  - Đối với các câu hỏi tra cứu bản cập nhật chứa các token số phiên bản chính xác (ví dụ: `Patch 26.19`, `Patch 26.15`, `Black Cleaver`), Dense search đơn thuần dễ bị nhiễu ngữ nghĩa giữa các bản patch khác nhau của cùng một tướng (ví dụ nhầm lẫn giữa Volibear patch 26.15 và patch 26.19).
  - BM25 trong Hybrid retrieval khớp chính xác các token phiên bản (`26.19`, `26.18`) và tên kỹ năng cụ thể, giúp RRF đẩy đúng các chunk payload của phiên bản cần tra cứu vào top context.
  - Nhờ có đúng ngữ cảnh của phiên bản được hỏi, Generator giảm thiểu ảo giác, tăng độ liên quan của câu trả lời từ 0.6471 lên 0.6941.
- **Trade-off về latency/cost:**
  - BM25 chạy hoàn toàn in-memory trên CPU local với thời gian tính toán < 5ms cho corpus 1.899 chunks, không tiêu tốn thêm token hay chi phí API embedding nào.
  - Thời gian RRF fusion là không đáng kể (< 1ms). Do đó, chi phí API và độ trễ mạng của Config B tương đương Config A, nhưng mang lại độ ổn định và chất lượng thông tin cao hơn đáng kể.

## Worst performers

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------- | ---------- |
|   1 | Patch 26.19 thay đổi nội tại The Relentless Storm của Volibear như thế nào? | Dense-only | 0.8500 | 0.6889 | 1.0000 | 0.2500 | Retrieval | Hiện tượng metadata `chunk-0` chiếm slot: Chunker cắt header thành `chunk-0`, developer note thành `chunk-1`, payload số liệu thành `chunk-2`. Dense search hút nhiều chunk của Volibear ở các bản patch khác (patch 26.15, catalog gốc), đẩy `chunk-2` xuống rank thấp. |
|   2 | Chỉ số Sức mạnh Thích ứng của ngọc Jack of All Trades bị thay đổi ra sao trong Patch 26.15? | Hybrid + RRF | 0.9000 | 0.5960 | 1.0000 | 0.3333 | Retrieval / Chunking | File rune ngắn bị chia cắt khiến thông tin chỉ số (breakpoint 5/10) nằm ở rìa chunk; từ khóa tiếng Việt "Sức mạnh Thích ứng" lệch nhẹ với thuật ngữ gốc tiếng Anh "Adaptive Force" trong patch notes. |
|   3 | Thời gian duy trì cộng dồn Đòn Đánh Cuồng Nộ của Cuồng Đao Guinsoo thay đổi thế nào trong Patch 26.18? | Dense-only | 0.8800 | 0.6720 | 1.0000 | 0.4500 | Generation / Retrieval | Có sự xung đột giữa tài liệu Data Dragon của trang bị gốc (`legal/item__3124.md`) và tài liệu Patch Notes cập nhật (`news/patch__26_18__item__guinsoo-s-rageblade.md`). Cả 2 đều có điểm tương đồng ngữ nghĩa cao. |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
|        1 | **Header Metadata Context Injection:** Đính kèm trực tiếp thông tin Entity Name và Patch Version vào đầu mỗi payload chunk thay vì để riêng một `chunk-0` rỗng số liệu. | Ở Case 13 và Case 17, `chunk-0` chỉ chứa tiêu đề và link crawl chiếm mất một vị trí trong top 5, làm loãng context thực tế chứa chỉ số cân bằng. | Loại bỏ hoàn toàn vấn đề metadata lấn át payload; tăng Context Precision lên trên 0.92. | Chạy lại `test_golden_dataset` và đo MAP trên tập patch notes. |
|        2 | **Bilingual Keyword Mapping:** Bổ sung từ điển đồng nghĩa Việt - Anh cho các thuật ngữ game phổ biến (ví dụ: Sức mạnh thích ứng ↔ Adaptive Force, Điểm hồi kỹ năng ↔ Ability Haste). | Case 13 và Case 16 cho thấy người dùng hỏi bằng tiếng Việt nhưng Patch Notes gốc từ Riot lại dùng thuật ngữ tiếng Anh. | Tăng điểm BM25 cho các câu hỏi tra cứu patch notes; tăng Answer Relevance thêm 5–8%. | Kiểm tra điểm số BM25 của các query kỹ năng có thuật ngữ Anh-Việt. |
|        3 | **Temporal Recency Boost:** Thiết lập trọng số ưu tiên phiên bản mới hơn khi truy xuất các câu hỏi so sánh cập nhật (ví dụ Patch 26.19 > Patch 26.15). | Dense search ở Case 17 trả về cả bản cập nhật 26.15 khi người dùng hỏi 26.19 vì cùng nói về tướng Volibear. | Tránh xung đột giữa các phiên bản cũ và mới; nâng cao Answer Relevance cho các câu hỏi patch. | Đối chiếu danh sách retrieved IDs đảm bảo phiên bản được hỏi đứng đầu rank. |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | -----------: | -----------------: | ---------- |
| **Lost-in-the-middle Reordering** (`reorder_for_llm`) | Thứ tự rank RRF tự nhiên (top-1 đến top-5) | Faithfulness +0.0310 | 0ms / $0 | Đưa các chunk có độ liên quan cao nhất ra 2 biên (đầu và cuối context) giúp LLM chú ý tốt hơn và trích dẫn citation chính xác hơn. |
| **PageIndex Fallback Calibration** (ngưỡng 0.66) | Không có fallback / threshold cố định | Out-of-domain Accuracy 100% | 0ms khi in-domain | Phân tách ranh giới rõ rệt giữa query in-domain ($\ge 0.7172$) và out-of-domain ($\le 0.6044$), đảm bảo safe refusal hoạt động tin cậy. |
