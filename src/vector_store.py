"""
Purpose:
    Convert a list of LangChain Document objects into vector embeddings
    using HuggingFace and store them locally in a FAISS index.
"""

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os

def save_to_faiss(docs: list[Document], index_dir="data/faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(index_dir)
