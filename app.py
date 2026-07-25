"""One-file Streamlit chat over the best-performing RAG config.

Run with: streamlit run app.py
"""
import streamlit as st

from raglab.config import CONFIGS, DEFAULT_CONFIG, GENERATION_MODEL, TOP_K
from raglab.generate import answer

st.set_page_config(page_title="FastAPI Docs RAG", page_icon="")
st.title("FastAPI Docs RAG")
st.caption(
    f"Local RAG over the FastAPI documentation - config `{DEFAULT_CONFIG}`, "
    f"top-{TOP_K} retrieval, {GENERATION_MODEL} via Ollama. "
    "The interesting part is the eval harness - see `docs/eval-report.md`."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for i, source in enumerate(message["sources"], start=1):
                    st.markdown(f"**[{i}] {source['file']}** (score {source['score']:.3f})")
                    st.caption(source["snippet"])

if question := st.chat_input("Ask something about FastAPI..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving and generating..."):
                text, chunks = answer(question, CONFIGS[DEFAULT_CONFIG])
        except RuntimeError as error:
            st.error(str(error))
        else:
            st.markdown(text)
            sources = [
                {"file": c.source_file, "score": c.score, "snippet": c.text[:300]}
                for c in chunks
            ]
            with st.expander("Sources"):
                for i, source in enumerate(sources, start=1):
                    st.markdown(f"**[{i}] {source['file']}** (score {source['score']:.3f})")
                    st.caption(source["snippet"])
            st.session_state.messages.append(
                {"role": "assistant", "content": text, "sources": sources}
            )
