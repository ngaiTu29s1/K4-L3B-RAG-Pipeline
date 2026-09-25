import streamlit as st
from dotenv import load_dotenv

from src.task10_generation import generate_with_citation


load_dotenv()

st.set_page_config(
    page_title="PatchLens",
    page_icon="⚔️",
    layout="wide",
)


def render_sources(sources: list[dict], retrieval_source: str) -> None:
    """Render source metadata and content below an answer."""
    if not sources:
        return
    st.caption(f"Retrieval: {retrieval_source}")
    for source in sources:
        metadata = source["metadata"]
        label = (
            f"{metadata['title']} · {source['retrieval_method']} · "
            f"{source['score']:.4f}"
        )
        with st.expander(label):
            st.caption(source["id"])
            if metadata.get("url"):
                st.markdown(f"[Mở nguồn gốc]({metadata['url']})")
            st.write(source["content"])


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("PatchLens")
    st.caption("Riot Data Dragon · Patch Notes chính thức")
    top_k = st.slider("Số nguồn truy xuất", 5, 10, 5)

st.title("⚔️ PatchLens")
st.caption("Hỏi về tướng, trang bị, phép bổ trợ và thay đổi trong Patch 26.15–26.19.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(
            message.get("sources", []),
            message.get("retrieval_source", "none"),
        )

query = st.chat_input("Ví dụ: Patch 26.19 thay đổi Volibear như thế nào?")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu nguồn..."):
            result = generate_with_citation(query, top_k=top_k)
        st.markdown(result["answer"])
        render_sources(result["sources"], result["retrieval_source"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "retrieval_source": result["retrieval_source"],
        }
    )
