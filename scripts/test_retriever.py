"""
Purpose:
    Test the document retrieval component by querying the FAISS index.
    Prints the top matching documents for a sample query.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retriever import load_vector_store, retrieve_documents

db = load_vector_store()  # Load the FAISS vector store
query = "What is LangChain?"
docs = retrieve_documents(db, query)  #  Pass both db and query

for i, doc in enumerate(docs):
    print(f"\n--- Document {i+1} ---\n")
    print(doc.page_content)
