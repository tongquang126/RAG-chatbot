"""
Purpose:
    Test the full RAG pipeline: retrieve documents and generate an answer.
"""

import sys, os
from dotenv import load_dotenv
load_dotenv()  # This ensures .env variables are loaded
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retriever import load_vector_store, retrieve_documents
from src.generator import generate_answer

db = load_vector_store()  # ✅ Load the FAISS index

query = "What is this document about?"
docs = retrieve_documents(db, query)  # Pass both db and query

print(generate_answer(docs, query))  # This order: (docs, query)
