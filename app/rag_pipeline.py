# app/rag_pipeline.py

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document


# 1. Load the HuggingFace embedding model
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# 2. Dummy document set for initial testing
def load_dummy_documents():
    docs = [
        Document(page_content="Changi Airport has four terminals: T1, T2, T3, and T4."),
        Document(page_content="Jewel Changi offers shopping, dining, and the Rain Vortex."),
        Document(page_content="Transit passengers can use lounges or book the YOTELAIR hotel."),
    ]
    return docs


# 3. Create vectorstore using FAISS
def create_vectorstore(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    embedding_model = get_embedding_model()
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    return vectorstore


# 4. Build the RAG chatbot pipeline using Ollama (Mistral)
def build_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = OllamaLLM(model="mistral", temperature=0.2)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return qa_chain


# 5. Expose chatbot for use in FastAPI or Streamlit
def get_chatbot():
    docs = load_dummy_documents()
    vectorstore = create_vectorstore(docs)
    qa = build_qa_chain(vectorstore)
    return qa
