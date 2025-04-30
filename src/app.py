# src/app.py
"""
Purpose:
    Launch a Gradio web interface for the RAG chatbot,
    allowing users to input questions and receive answers
    using the FAISS index and GPT-3.5.
"""
import gradio as gr
from retriever import load_vector_store, retrieve_documents
from generator import generate_answer
from dotenv import load_dotenv
load_dotenv()  # This ensures .env variables are loade
import os

load_dotenv()  # Load biến môi trường từ .env

db = load_vector_store()

def chatbot(query):
    docs = retrieve_documents(db, query)
    answer = generate_answer(docs, query)  # Không cần truyền API key nữa
    return answer

iface = gr.Interface(
    fn=chatbot,
    inputs="text",
    outputs="text",
    title="RAG Chatbot",
    description="Enter a question"
)

if __name__ == "__main__":
    iface.launch()
