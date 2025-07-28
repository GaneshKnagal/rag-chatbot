import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import weaviate
from weaviate.classes.config import Configure, Property, DataType

# ---- STEP 1: Load your guidebook text ----
def load_guidebook_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# ---- STEP 2: Split into chunks ----
def chunk_text(text, chunk_size=512, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

# ---- STEP 3: Connect to Weaviate v4 ----
def connect_to_weaviate():
    client = weaviate.connect_to_local()  # assumes running at localhost:8080
    return client

# ---- STEP 4: Create class if not exists ----
def ensure_class_exists(client):
    if "GuidebookChunk" not in client.collections.list_all():
        client.collections.create(
            name="GuidebookChunk",
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="content", data_type=DataType.TEXT),
            ],
        )

# ---- STEP 5: Store chunks with embeddings ----
def store_chunks_in_weaviate(client, chunks, embeddings):
    collection = client.collections.get("GuidebookChunk")
    for chunk in chunks:
        vector = embeddings.embed_query(chunk)
        collection.data.insert(
            properties={"content": chunk},
            vector=vector,
        )

# ---- MAIN RUN ----
if __name__ == "__main__":
    print("📄 Loading guidebook...")
    file_path = "data/raw/changi_guidebook.txt"
    text = load_guidebook_text(file_path)

    print("🔪 Chunking...")
    chunks = chunk_text(text)

    print("🔌 Connecting to Weaviate...")
    client = connect_to_weaviate()

    print("📁 Ensuring class exists...")
    ensure_class_exists(client)

    print(f"📥 Storing {len(chunks)} chunks...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    store_chunks_in_weaviate(client, chunks, embeddings)

    print("✅ All chunks stored in Weaviate.")
