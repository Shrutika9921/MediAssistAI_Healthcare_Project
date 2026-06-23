"""
MediAssist AI — Streamlit Chat Interface
A healthcare document assistant with chat UI, document upload, and file management.
Uses FAISS for vector storage.

Usage:
    streamlit run UI/streamlit_app.py
"""

import streamlit as st
import sys
import os
import requests
import importlib

# FORCE CACHE CLEAR to fix the langgraph state_modifier bug
st.cache_resource.clear()

# ─── Path Setup ──────────────────────────────────────────
# Add App directory to path for imports
APP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "App")
sys.path.insert(0, APP_DIR)

import importlib
from backend.config import UPLOAD_DIR, VECTOR_DB_DIR
from document_loader import load_documents
from chunking import create_chunks
from embedding_service import get_embedding_model
from vector_store import create_vector_store, load_vector_store
from retriever import get_retriever, retrieve_documents

from rag_chain import build_rag_chain, ask_question


# ─── Page Config ─────────────────────────────────────────
st.set_page_config(
    page_title="MediAssistAI",
    layout="wide"
)

# ─── Minimal Styling ─────────────────────────────────────
st.markdown("""
<style>
.main-header{
    position:sticky;
    top:0;
    background:#000000;
    z-index:999;
    padding:10px;
    border-bottom:1px solid #333;
    color: #ffffff;
}
.source{
    font-size:12px;
    color:gray;
    margin-top:10px;
}
.file-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid #eee;
}
.stButton > button {
    margin: 0;
}
/* ── Right Aside Panel ── */
.right-aside {
    position: fixed;
    top: 70px;
    right: 0;
    width: 280px;
    height: 100vh;
    background-color: #0e0e0e;
    border-left: 1px solid #333;
    padding: 20px;
    z-index: 900;
    overflow-y: auto;
    color: #fff;
    font-size: 14px;
}
.right-aside h3 {
    font-size: 16px;
    border-bottom: 1px solid #333;
    padding-bottom: 10px;
    margin-bottom: 15px;
}
.right-aside ul {
    list-style-type: none;
    padding: 0;
    margin: 0;
}
.right-aside li {
    background: #1a1a1a;
    padding: 10px;
    margin-bottom: 8px;
    border-radius: 6px;
    border: 1px solid #2a2a2a;
    line-height: 1.4;
}
/* Ensure main content doesn't overlap with right aside */
.block-container {
    padding-right: 300px !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Header ──────────────────────────────────────────────
st.markdown("""
<div class="main-header">
<h1>🏥 MediAssistAI</h1>
</div>
""", unsafe_allow_html=True)


# ─── Session State ───────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "question_history" not in st.session_state:
    st.session_state.question_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False


# ─── Helper Functions ────────────────────────────────────
# Trigger auto-reload cache clear
@st.cache_resource
def init_rag_pipeline():
    """Initialize the FAISS RAG pipeline."""
    embeddings = get_embedding_model()
    vector_store = load_vector_store(embeddings)

    if vector_store is None:
        return None, None

    retriever = get_retriever(vector_store)
    rag_chain = build_rag_chain(retriever)
    return rag_chain, retriever

def reindex_documents():
    """Run ingestion pipeline on current files in UPLOAD_DIR and reinit."""
    embeddings = get_embedding_model()
    docs = load_documents(UPLOAD_DIR)
    
    if not docs:
        st.session_state.pipeline_ready = False
        return False
        
    chunks = create_chunks(docs)
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    create_vector_store(chunks, embeddings)
    
    init_rag_pipeline.clear()
    rag_chain, retriever = init_rag_pipeline()
    st.session_state.rag_chain = rag_chain
    st.session_state.retriever = retriever
    st.session_state.pipeline_ready = True
    return True

# Initialize pipeline on startup
if not st.session_state.pipeline_ready:
    rag_chain, retriever = init_rag_pipeline()
    if rag_chain:
        st.session_state.rag_chain = rag_chain
        st.session_state.retriever = retriever
        st.session_state.pipeline_ready = True


# ─── Sidebar ─────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Document")
    
    uploaded_file = st.file_uploader(
        "Upload PDF / DOCX / TXT",
        type=["pdf", "docx", "txt"],
        key="file_uploader_widget"
    )

    if uploaded_file:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"{uploaded_file.name} uploaded successfully")
        
        # Trigger re-indexing
        with st.spinner("Indexing new document..."):
            if reindex_documents():
                st.success("Re-indexing complete!")
            else:
                st.error("Failed to re-index.")

    st.divider()

    st.header("📑 Uploaded Documents")
    
    if os.path.exists(UPLOAD_DIR):
        files = [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]
        
        if files:
            for file in files:
                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    st.write(f"📄 {file}")
                with col2:
                    if st.button("❌", key=f"del_{file}", help="Delete this file"):
                        # Delete file logic
                        file_path = os.path.join(UPLOAD_DIR, file)
                        try:
                            os.remove(file_path)
                            st.success(f"Deleted {file}")
                            # Re-index
                            with st.spinner("Updating index..."):
                                reindex_documents()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting file: {e}")
        else:
            st.info("No documents uploaded")
    else:
        st.info("No documents uploaded")


# ─── Chat History (Main Layout) ──────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)


# ─── Chat Input ──────────────────────────────────────────
st.divider()

query = st.chat_input("Ask your medical question...")

if query:
    st.session_state.question_history.append(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        if not st.session_state.pipeline_ready:
            answer = "No documents found. Please upload a document in the sidebar to ask questions."
            st.markdown(answer)
        else:
            with st.spinner("Thinking..."):
                try:
                    answer = ask_question(st.session_state.rag_chain, query)
                    source_docs = retrieve_documents(st.session_state.retriever, query)
                    
                    sources = []
                    for doc in source_docs:
                        source_name = doc.metadata.get("source_filename", "Unknown")
                        if source_name not in sources:
                            sources.append(source_name)

                    if sources:
                        answer += f"<div class='source'>(Source: {', '.join(sources)})</div>"
                    else:
                        answer += "<div class='source'>(Source: Not Available)</div>"
                        
                except Exception as e:
                    answer = f"Error generating answer: {e}"

                st.markdown(answer, unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ─── Recent Chats (Fixed Right Aside) ────────────────────
# Rendered at the very end so it includes queries made in the current run
recent_html = "<div class='right-aside'><h3>🕒 Recent Chats</h3><ul>"
if st.session_state.question_history:
    for q in reversed(st.session_state.question_history[-10:]):
        recent_html += f"<li>{q}</li>"
else:
    recent_html += "<li><span style='color:gray;'>No questions yet.</span></li>"
recent_html += "</ul></div>"

st.markdown(recent_html, unsafe_allow_html=True)
