# app/main.py

from fastapi import FastAPI, Query
from app.rag_pipeline import get_chatbot

app = FastAPI()
chatbot = get_chatbot()

@app.get("/ask")
def ask(query: str = Query(..., description="Ask a question about Changi or Jewel Airport")):
    response = chatbot.run(query)
    return {"query": query, "answer": response}
