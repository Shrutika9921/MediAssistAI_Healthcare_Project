"""
MediAssist AI — Vector Store
Creates and loads a ChromaDB persistent vector store for document embeddings.
"""

import os
import shutil
from langchain_chroma import Chroma

# Import config
try:
    from backend.config import VECTOR_DB_DIR, CHROMA_COLLECTION_NAME
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from backend.config import VECTOR_DB_DIR, CHROMA_COLLECTION_NAME


def create_vector_store(chunks, embeddings):
    """
    Create a new ChromaDB vector store from document chunks.

    If a vector store already exists at the persist directory,
    it will be cleared and recreated.

    Args:
        chunks: List of LangChain Document objects (chunked).
        embeddings: HuggingFace embedding model instance.

    Returns:
        Chroma vector store instance.
    """
    # Clear existing vector store if it exists
    if os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
        print(f"  [INFO] Clearing existing vector store at {VECTOR_DB_DIR}")
        shutil.rmtree(VECTOR_DB_DIR)
        os.makedirs(VECTOR_DB_DIR, exist_ok=True)

    print(f"  [INFO] Creating vector store with {len(chunks)} chunks...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=VECTOR_DB_DIR
    )

    print(f"  [OK] Vector store created and persisted at {VECTOR_DB_DIR}")
    return vector_store


def load_vector_store(embeddings):
    """
    Load an existing ChromaDB vector store from disk.

    Args:
        embeddings: HuggingFace embedding model instance.

    Returns:
        Chroma vector store instance, or None if not found.
    """
    if not os.path.exists(VECTOR_DB_DIR) or not os.listdir(VECTOR_DB_DIR):
        print(f"  [ERROR] No vector store found at {VECTOR_DB_DIR}")
        print("  [INFO] Run 'python ingest.py' first to create the vector store.")
        return None

    print(f"  [INFO] Loading vector store from {VECTOR_DB_DIR}")

    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_DIR
    )

    # Verify the collection has documents
    count = vector_store._collection.count()
    print(f"  [OK] Vector store loaded with {count} documents")

    return vector_store
