"""
MediAssist AI — Centralized Configuration
All RAG pipeline settings in one place.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file (at project root)
_ENV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    ".env"
)
load_dotenv(_ENV_PATH)

# ─── Paths ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOAD_DIR = os.path.join(BASE_DIR, "Data", "uploads")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "Data", "vector_db")

# ─── Embedding Model ────────────────────────────────────
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# ─── Chunking Settings ──────────────────────────────────
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

# ─── ChromaDB Settings ──────────────────────────────────
CHROMA_COLLECTION_NAME = "mediassist_docs"

# ─── Groq LLM Settings ──────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
GROQ_TEMPERATURE = 0.1
GROQ_MAX_TOKENS = 1024

# ─── Retriever Settings ─────────────────────────────────
RETRIEVER_TOP_K = 5
RETRIEVER_SCORE_THRESHOLD = 0.3
