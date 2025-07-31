import sys
import os
import streamlit as st
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.rag_pipeline import get_chatbot

# 🚀 Page Configuration
st.set_page_config(page_title="Changi RAG Chatbot", layout="wide")
st.title("🛫 Changi & Jewel Airport Chatbot")
st.markdown("Ask anything about Changi Airport, Jewel, facilities, shops, lounges, and more.")

# ⚙️ Sidebar Options
st.sidebar.header("🔧 Settings")
render_markdown = st.sidebar.checkbox("Render Answer as Markdown", value=True)
show_full_chunks = st.sidebar.checkbox("Show Full Retrieved Chunks", value=False)

# 🔄 Load RAG Chain
qa_chain = get_chatbot()

# 💬 User Query
query = st.text_input("💬 Ask your question here:")

if query:
    with st.spinner("🔍 Searching and generating answer..."):
        start_time = time.time()

        # Run the RAG pipeline
        try:
            result = qa_chain.invoke({"query": query}, return_only_outputs=False)
            answer = result.get("result", "No answer generated.")
            source_docs = result.get("source_documents", [])
        except Exception as e:
            answer = "❌ Error generating answer."
            source_docs = []
            st.error(f"Exception: {e}")

        end_time = time.time()

    # 💡 Answer Section
    st.subheader("💡 Answer")
    if render_markdown:
        st.markdown(answer)
    else:
        st.success(answer)

    # 🧰 Debug Info
    st.markdown("---")
    st.subheader("🛠️ Debug Info")
    st.markdown(f"**Query:** `{query}`")
    st.markdown(f"**Time Taken:** `{round(end_time - start_time, 2)} seconds`")
    st.markdown("**LLM Used:** `Claude 3 Sonnet via AWS Bedrock`")
    st.markdown(f"**Retrieved Chunks:** `{len(source_docs)}`")

    # 📄 Source Chunks
    if source_docs:
        st.subheader("📄 Source Documents")
        for i, doc in enumerate(source_docs):
            with st.expander(f"Chunk {i+1}"):
                if show_full_chunks:
                    st.markdown(doc.page_content)
                else:
                    st.code(doc.page_content.strip()[:1000], language="markdown")
    else:
        st.info("No source documents retrieved.")
