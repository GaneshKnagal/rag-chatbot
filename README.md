
# 🤖 RAG Chatbot with Claude 3 Sonnet (via Amazon Bedrock)

This project is a **Retrieval-Augmented Generation (RAG)** based chatbot that answers user queries using local document embeddings and Amazon Bedrock's Claude 3 Sonnet model.

It includes:
- 📄 Document embedding & chunk retrieval using Pinecone
- 🤝 LLM interaction via Amazon Bedrock
- ⚙️ Orchestration using LangChain
- 🖥️ Deployed on Streamlit Cloud with Streamlit UI

---

## 🚀 Live Demo

🔗 https://appchatbotpy-erqfvwwsfhcktmbdckxvbu.streamlit.app/

---

## 🛠️ Tech Stack

| Layer         | Technology         |
|---------------|--------------------|
| UI            | Streamlit          |
| Backend       | Python, LangChain  |
| LLM Provider  | Claude 3 Sonnet (via Amazon Bedrock) |
| Vector Store  | Pinecone           |
| Embeddings    | HuggingFace (MiniLM) |
| Deployment    | Streamlit Cloud    |

---

## 📁 Project Structure

```
chatbot_rag_project/
├── app/
│   ├── rag_pipeline.py             # RAG logic using LangChain
│   ├── test_chat.py                 # Test RAG queries
│   ├── ingest_local_to_pinecone.py    # Direct Claude API test
│   └── ...
├── ui/
│   └── streamlit_chatbot.py    # Streamlit UI
├── .env                        # API Keys and Bedrock settings
├── example.txt
├── requirements.txt
└── README.md
```

---

## 🔐 .env Configuration

Create a `.env` file in the root with:

```env
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENV=your_env
PINECONE_INDEX_NAME=your_index
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_DEFAULT_REGION=us-east-1
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
```

---

## ⚙️ Run Locally

1. Clone repo:

```bash
git clone https://github.com/YourUsername/rag-chatbot.git
cd rag-chatbot
```

2. Create virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Set `.env` and run:

```bash
streamlit run ui/streamlit_chatbot.py
```

---

## 📌 Deployment

1. Deployed on **Streamlit Cloud**
2. Code pushed to GitHub and connected to Streamlit Cloud app
3. Environment variables managed via Streamlit Secrets

---

## 📚 Features

- Ask domain-specific questions (e.g., airport FAQs, documents, etc.)
- Answers are context-aware from your custom documents
- Claude 3 Sonnet ensures high-quality, coherent responses

---

## 🙋‍♂️ Author

👤 **Ganesh Kumar Nagal**  
📫 [ganeshnagal55@gmail.com](mailto:ganeshnagal55@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/ganesh-kumar-nagal-5430bb1b6/)  
🔗 [GitHub](https://github.com/GaneshKnagal)

---

## 📝 License

MIT License. See `LICENSE` for details.
