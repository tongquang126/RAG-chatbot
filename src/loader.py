"""
Purpose:
    Load PDF and TXT documents from a folder and split them into smaller chunks
    for embedding and retrieval using LangChain's text splitter.
"""

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_and_split_documents(folder_path="data/documents"):
    docs = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())

        elif filename.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")  # đảm bảo UTF-8
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)
