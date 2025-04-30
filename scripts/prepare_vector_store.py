"""
Purpose:
    Load and split PDF documents from data/documents/,
    generate vector embeddings using HuggingFace,
    and save them as a FAISS index to data/faiss_index/.
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.loader import load_and_split_documents
from src.vector_store import save_to_faiss

docs = load_and_split_documents()
save_to_faiss(docs)
print("FAISS index created successfully.")
