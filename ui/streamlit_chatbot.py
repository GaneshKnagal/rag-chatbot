import streamlit as st
from app.rag_pipeline import get_chatbot
import time

# Set page config
st.set_page_config(page_title="Changi RAG Chatbot", layout="wide")

# Title
st.title("🛫 Changi & Jewel Airport Chatbot")
st.markdown("Ask anything about Changi Airport, Jewel, facilities, shops, lounges, and more.")

# Get chatbot chain
qa_chain = get_chatbot()

# Input field
query = st.text_input("Ask your question here:", "")

if query:
    with st.spinner("🔍 Searching and generating answer..."):
        start_time = time.time()

        # Run chain with source documents returned
        result = qa_chain.invoke(query, return_only_outputs=False)

        answer = result['result']
        retrieved_docs = result['intermediate_steps'][0].documents if 'intermediate_steps' in result else []

        end_time = time.time()

        # Show answer
        st.subheader("💬 Answer:")
        st.write(answer)

        # Debug info
        st.markdown("---")
        st.subheader("🧠 Debug Information")

        st.markdown(f"**Query:** `{query}`")
        st.markdown(f"**Time Taken:** `{round(end_time - start_time, 2)} seconds`")
        st.markdown("**LLM Used:** `Mistral via Ollama`")
        st.markdown(f"**Chunks Retrieved:** `{len(retrieved_docs)}`")

        # Show retrieved content
        st.subheader("📄 Retrieved Documents:")
        for i, doc in enumerate(retrieved_docs):
            st.markdown(f"**Document {i+1}:**")
            st.code(doc.page_content.strip()[:1000])  # Limit display to 1000 chars

