import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def store_embeddings():
    print("🔄 Loading documents...")
    loader = TextLoader(
        r"C:\Users\ganes\Downloads\chatbot_rag_project\data\changi_guidebook.txt", 
        encoding="utf-8"
    )  
    raw_docs = loader.load()

    print("✂️ Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,          # reduced from 1000
        chunk_overlap=100,
        length_function=len
    )
    docs = text_splitter.split_documents(raw_docs)

    # 🛠️ Trim metadata to avoid size overflow
    for doc in docs:
        doc.metadata = {"source": doc.metadata.get("source", "local")}

    print(f"📚 Total Chunks Created: {len(docs)}")

    print("🧠 Loading local embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("🪵 Storing to Pinecone...")
    vectorstore = PineconeVectorStore.from_documents(
        docs,
        embedding=embeddings,
        index_name=os.getenv("PINECONE_INDEX_NAME")
    )

    print("✅ Embedding and storage complete!")

if __name__ == "__main__":
    store_embeddings()
