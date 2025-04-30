"""
Purpose:
    Load the FAISS vector store from disk and retrieve the most relevant documents
    for a given query using similarity search.
"""

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

def load_vector_store(index_dir: str = "data/faiss_index") -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    return db

def retrieve_documents(db: FAISS, query: str, k: int = 3) -> list[Document]:
    return db.similarity_search(query, k=k)
