import os
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_aws import ChatBedrock
from pinecone import Pinecone

# Load .env variables
load_dotenv()

# --------- Prompt Template ----------
def get_prompt_template():
    template = """
Answer the question based on the context below. Be precise and concise.

Context:
{context}

Question:
{question}
"""
    return PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

# --------- Load Embedding Model ----------
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --------- Load from Pinecone (already embedded) ----------
def get_vectorstore():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "rag-chatbot-index"
    index = pc.Index(index_name)
    return PineconeVectorStore(
        index=index,
        embedding=get_embedding_model(),
        text_key="text"
    )

# --------- Load Claude 3 via Bedrock ----------
def get_llm():
    return ChatBedrock(
        model_id=os.getenv("BEDROCK_MODEL_ID"),  # e.g., "anthropic.claude-3-sonnet-20240229-v1:0"
        region_name=os.getenv("AWS_DEFAULT_REGION")
    )

# --------- Final RAG Chain (retrieval + LLM) ----------
def get_chatbot():
    vectorstore = get_vectorstore()
    prompt = get_prompt_template()
    llm = get_llm()

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True  # Optional: useful for debug panel
    )
