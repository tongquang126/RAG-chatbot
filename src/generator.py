"""
Purpose:
    Generate an answer to a user query using GPT-3.5,
    by incorporating retrieved document content as context.
"""

from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from typing import List
import os

def generate_answer(docs: List[Document], query: str) -> str:
    # API key đã được load từ .env nên không cần truyền tham số
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    return llm.predict(prompt)
