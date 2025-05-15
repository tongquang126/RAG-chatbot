# 🤖 RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that combines OpenAI's GPT-3.5/4 with your internal PDF documents to generate more accurate and context-aware responses.

---

## 📌 Purpose

This project enhances Large Language Models (LLMs) by enabling them to answer queries based on **private documents** rather than relying solely on general internet knowledge.

🔍 It works by:
- Retrieving relevant information from internal PDF documents.
- Feeding that information to OpenAI's GPT model to generate answers grounded in your content.

This approach improves reliability and reduces hallucinations in LLM outputs for domain-specific use cases.

---

## 🛠️ Features

- ✅ Ingests and processes PDF documents.
- ✅ Embeds and indexes text using HuggingFace + FAISS.
- ✅ Retrieves top relevant chunks per query.
- ✅ Generates contextual answers via OpenAI's GPT-3.5/4.
- ✅ Simple chatbot interface powered by Gradio.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/tongquang126/RAG-chatbot.git
cd RAG-chatbot
2. Set up environment using Pipenv
Make sure you have Pipenv installed:

pip install pipenv
Then install dependencies and activate the virtual environment:

pipenv install
pipenv shell
3. Add OpenAI API key
Create a .env file in the root directory with your OpenAI key:
OPENAI_API_KEY=your_openai_key_here

4. Add PDF documents
Place your .pdf files into the data/documents/ directory.

5. Build the vector store
Run this script to process and index documents:
python scripts/prepare_vector_store.py

6. Launch the chatbot interface
Start the Gradio app:
python src/app.py

🧪 Testing
Test document retrieval:
python scripts/test_retriever.py

Test RAG pipeline (retriever + generator):
python scripts/test_generator.py

⚙️ Tech Stack
Python
LangChain
OpenAI API
FAISS
HuggingFace Transformers
Gradio
Pipenv

📄 License
MIT License. Feel free to use, modify, and distribute this project.
