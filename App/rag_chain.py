"""
MediAssist AI — RAG Chain
Connects the retriever to Groq LLM with a healthcare-specific prompt.
Returns answers grounded in retrieved context with source citations.
"""

import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Import config
try:
    from backend.config import GROQ_API_KEY, GROQ_MODEL_NAME, GROQ_TEMPERATURE, GROQ_MAX_TOKENS
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from backend.config import GROQ_API_KEY, GROQ_MODEL_NAME, GROQ_TEMPERATURE, GROQ_MAX_TOKENS


# ─── Healthcare-Specific System Prompt ───────────────────
SYSTEM_PROMPT = """You are MediAssist AI, an intelligent healthcare information assistant.
Your role is to provide accurate, helpful answers based ONLY on the provided context from hospital documents.

RULES:
1. Answer ONLY based on the retrieved context below. Do NOT use your general knowledge.
2. If the context does not contain enough information to answer the question, clearly state:
   "I don't have enough information in the available documents to answer this question."
3. Always cite the source document(s) in your answer by mentioning the filename.
4. Be precise and professional. Use medical terminology appropriately.
5. If the question is about a specific procedure, protocol, or guideline, quote the relevant section.
6. Structure your response clearly with bullet points or numbered steps when appropriate.

CONTEXT FROM HOSPITAL DOCUMENTS:
{context}
"""

USER_PROMPT = """Question: {question}

Provide a detailed, well-structured answer based on the context above."""


def _format_docs(docs):
    """Format retrieved documents into a single context string with source info."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_filename", "Unknown")
        page = doc.metadata.get("page", "N/A")
        formatted.append(
            f"[Source {i}: {source} | Page: {page}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)


def get_llm():
    """
    Initialize the Groq LLM.

    Returns:
        ChatGroq LLM instance.
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY not found! Please add it to your .env file:\n"
            "GROQ_API_KEY=gsk_your_key_here"
        )

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL_NAME,
        temperature=GROQ_TEMPERATURE,
        max_tokens=GROQ_MAX_TOKENS
    )

    return llm


def build_rag_chain(retriever):
    """
    Build the complete RAG chain: Retriever → Prompt → LLM → Output.

    Args:
        retriever: LangChain retriever instance.

    Returns:
        A runnable RAG chain that accepts a question string
        and returns an answer string.
    """
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT)
    ])

    rag_chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_question(rag_chain, question):
    """
    Ask a question to the RAG chain and get an answer.

    Args:
        rag_chain: The built RAG chain.
        question: User's question string.

    Returns:
        Answer string from the LLM.
    """
    answer = rag_chain.invoke(question)
    return answer
