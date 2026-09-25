import os
import re
import streamlit as st
from dotenv import load_dotenv

from src.task10_generation import generate_with_citation


load_dotenv()

st.set_page_config(
    page_title="PatchLens",
    page_icon="⚔️",
    layout="wide",
)


def render_sources(sources: list[dict], retrieval_source: str, answer: str = "") -> None:
    """Render source metadata and content below an answer, separating cited sources from candidate context."""
    if not sources:
        return

    st.caption(f"🔍 **Phương thức truy xuất:** `{retrieval_source.upper()}`")

    # Extract cited IDs from answer (e.g. [news/patch__26_19...::chunk-1])
    cited_ids = set(re.findall(r"\[([^\[\]]+)\]", answer)) if answer else set()
    cited_sources = [s for s in sources if s.get("id") in cited_ids]
    other_sources = [s for s in sources if s.get("id") not in cited_ids]

    # 1. Directly Cited Sources (Prominently displayed)
    if cited_sources:
        st.markdown("##### 🎯 Nguồn trích dẫn trực tiếp trong câu trả lời:")
        for idx, source in enumerate(cited_sources, start=1):
            metadata = source.get("metadata", {})
            title = metadata.get("title", "Tài liệu")
            chunk_id = source.get("id", "")
            method = source.get("retrieval_method", "hybrid").upper()
            score = source.get("score", 0.0)
            url = metadata.get("url")

            with st.expander(f"📌 [{idx}] {title} · {method} (Score: {score:.4f})", expanded=True):
                st.markdown(f"**Chunk ID:** `{chunk_id}`")
                if url:
                    st.markdown(f"🔗 [Mở bài viết gốc trên Riot Games]({url})")
                st.markdown("---")
                st.markdown(source.get("content", ""))

    # 2. Other candidate chunks in Top-K (Collapsed by default so they don't cause confusion)
    if other_sources:
        with st.expander(f"📁 Xem thêm {len(other_sources)} ngữ cảnh liên quan khác trong Top {len(sources)} (Không dùng)", expanded=False):
            st.caption(
                "Các đoạn tài liệu dưới đây được thuật toán tìm kiếm trả về trong Top-K ứng viên, "
                "nhưng mô hình LLM đã chọn lọc và không sử dụng trực tiếp để đưa vào câu trả lời."
            )
            for idx, source in enumerate(other_sources, start=1):
                metadata = source.get("metadata", {})
                title = metadata.get("title", "Tài liệu")
                chunk_id = source.get("id", "")
                method = source.get("retrieval_method", "hybrid").upper()
                score = source.get("score", 0.0)
                url = metadata.get("url")

                st.markdown(f"**[{idx}] {title}** · `{method}` (Score: `{score:.4f}`)")
                st.caption(f"ID: `{chunk_id}`")
                if url:
                    st.markdown(f"🔗 [Mở nguồn gốc]({url})")
                st.markdown(f"> {source.get('content', '')[:250]}...")
                st.markdown("---")


if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar setup
with st.sidebar:
    st.title("PatchLens")
    st.caption("Riot Data Dragon · Patch Notes 26.15 – 26.19")
    top_k = st.slider("Số nguồn truy xuất (Top K)", 5, 10, 5)

    if st.button("🗑️ Xóa lịch sử trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("### 💡 Gợi ý tra cứu nhanh")
    sample_queries = [
        "Patch 26.19 thay đổi nội tại The Relentless Storm của Volibear như thế nào?",
        "Trang bị Vô Cực Kiếm cung cấp những chỉ số gì và giá mua là bao nhiêu?",
        "Patch 26.15 thay đổi chiêu Q Thundering Smash của Volibear như thế nào?",
        "Hồi chiêu Dịch Chuyển (Teleport) đường trên thay đổi thế nào trong Patch 26.19?",
        "Công thức nấu phở bò Hà Nội chuẩn vị truyền thống?",
    ]

    selected_sample = None
    for sq in sample_queries:
        if st.button(sq, key=f"sample_{sq}", use_container_width=True):
            selected_sample = sq

    st.markdown("---")
    st.markdown("### ⚙️ Thông số hệ thống")
    st.markdown(
        "- **Retrieval:** Hybrid (Dense + BM25 + RRF)\n"
        "- **Vector DB:** ChromaDB (1.899 chunks)\n"
        "- **LLM:** `gemini-3.5-flash-lite`\n"
        "- **Fallback threshold:** 0.66 (PageIndex)\n"
        "- **Safety:** Grounded citation & safe refusal"
    )

st.title("⚔️ PatchLens")
st.caption("Trợ lý thông minh hỏi đáp dữ liệu Liên Minh Huyền Thoại: Tướng, Trang bị, Phép bổ trợ và Bản cập nhật 26.15–26.19.")

# Render existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(
                message.get("sources", []),
                message.get("retrieval_source", "none"),
                answer=message.get("content", ""),
            )

# Handle user input (from chat box or quick sample button)
input_query = st.chat_input("Ví dụ: Patch 26.19 thay đổi Volibear như thế nào?")
query = input_query or selected_sample

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu nguồn và sinh câu trả lời có trích dẫn..."):
            result = generate_with_citation(query, top_k=top_k)
        st.markdown(result["answer"])
        render_sources(result["sources"], result["retrieval_source"], answer=result["answer"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "retrieval_source": result["retrieval_source"],
        }
    )
    st.rerun()
