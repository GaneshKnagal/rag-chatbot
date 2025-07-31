from rag_pipeline import get_chatbot

if __name__ == "__main__":
    chatbot = get_chatbot()

    question = "What is the immigration process at Changi?"
    print("❓ Question:", question)

    response = chatbot.invoke(question)
    print("🤖 Answer:", response)



