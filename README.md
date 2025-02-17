RAG Chatbot for My Portfolio 🚀

This is a Retrieval-Augmented Generation (RAG) chatbot designed to answer questions based on my portfolio, projects, and education. The chatbot leverages Hugging Face for embeddings, FAISS for efficient vector search, and Chat Groq (LLaMA-3.3-70B-Versatile) as the LLM for generating responses. The frontend is built using Streamlit, making it easy to interact with the bot.

Features ✨

📄 Retrieves answers from my portfolio, projects, and education data

🧠 Uses FAISS for fast and scalable vector search

🔗 LangChain-powered retrieval for better contextual responses

🤖 Embeds text with Hugging Face models

🏎️ Generates responses using Chat Groq (LLaMA-3.3-70B-Versatile)

🌐 User-friendly web interface with Streamlit

🚀 Easily deployable on Streamlit Cloud or other hosting services

Tech Stack 🛠

Embeddings: Hugging Face (sentence-transformers/all-mpnet-base-v2)

Vector Database: FAISS

LLM: Chat Groq (llama-3.3-70b-versatile)

Frontend & Deployment: Streamlit

Note 📝

If you want to clone this application, you must create a vector database using your own portfolio, projects, and education details. The chatbot retrieves answers based on the embedded data, so you need to:
Replace my vector_database folder with your own.
Generate new embeddings using Hugging Face models.
Create a FAISS vector database from your data before running the chatbot.
Make sure to update the paths and configurations accordingly! 🚀
