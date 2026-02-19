
import streamlit as st
import requests
import os

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "user_session_001"
if "ingestion_done" not in st.session_state:
    st.session_state.ingestion_done = False
if "ingestion_stats" not in st.session_state:
    st.session_state.ingestion_stats = {}


def check_backend() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def get_status() -> dict:
    try:
        r = requests.get(f"{API_URL}/status", timeout=5)
        return r.json()
    except Exception:
        return {}


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.title("🧠 RAG Knowledge Assistant")
    st.markdown("---")

    # Backend status
    backend_ok = check_backend()
    if backend_ok:
        st.success("✅ Backend connected")
    else:
        st.error(f"❌ Backend unreachable at {API_URL}")
        st.caption("Ensure the FastAPI container is running.")

    st.markdown("---")

    st.subheader("📁 Upload Documents")
    st.caption("Supports: PDF, DOCX, PPTX, TXT, MD, CSV, JSON, XML, SQLite")

    uploaded_files = st.file_uploader(
        "Upload your documents",
        accept_multiple_files=True,
        type=["pdf", "txt", "md", "csv", "json", "xml", "docx", "pptx", "db", "sqlite"],
        label_visibility="collapsed"
    )

    if uploaded_files and backend_ok:
        if st.button("🚀 Ingest Documents", use_container_width=True, type="primary"):
            with st.spinner(f"Processing {len(uploaded_files)} file(s)..."):
                try:
                    files_payload = [
                        ("files", (f.name, f.read(), f.type or "application/octet-stream"))
                        for f in uploaded_files
                    ]
                    r = requests.post(
                        f"{API_URL}/ingest",
                        files=files_payload,
                        timeout=120
                    )
                    if r.status_code == 200:
                        result = r.json()
                        st.session_state.ingestion_done = True
                        st.session_state.ingestion_stats = result
                        st.success(f"✅ Ingested {result['doc_count']} documents")
                        st.rerun()
                    else:
                        st.error(f"Ingestion failed: {r.json().get('detail', r.text)}")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")

    if st.session_state.ingestion_done:
        st.subheader("📊 Knowledge Base")
        stats = st.session_state.ingestion_stats
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", stats.get("doc_count", 0))
            st.metric("Chunks", stats.get("chunk_count", 0))
        with col2:
            st.metric("Vectors", stats.get("vector_count", 0))
            st.metric("File Types", len(stats.get("file_types", {})))

        if stats.get("file_types"):
            st.caption("File types loaded:")
            for ftype, count in stats["file_types"].items():
                st.caption(f"  • {ftype}: {count} docs")

    st.markdown("---")

    st.subheader("🔄 Session")
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        try:
            requests.delete(
                f"{API_URL}/session/{st.session_state.session_id}",
                timeout=5
            )
        except Exception:
            pass
        st.rerun()

    status = get_status()
    if status:
        st.caption(f"Model: {status.get('llm_model', 'N/A')}")
        st.caption(f"Embeddings: all-MiniLM-L6-v2")


# ==========================================
# MAIN CHAT
# ==========================================

st.title("💬 Intelligent Knowledge Assistant")

if not backend_ok:
    st.error("Backend service is not running. Start the FastAPI container.")
    st.stop()

if not st.session_state.ingestion_done:
    st.info("👈 Upload and ingest documents using the sidebar to begin querying.")
    with st.expander("💡 What can this assistant do?"):
        st.markdown("""
        This RAG-powered assistant answers questions from:
        - 📄 **PDF, Word, PowerPoint** documents
        - 📝 **Text and Markdown** notes
        - 📊 **CSV datasets**
        - 🔧 **JSON and XML** data files
        - 🗄️ **SQLite databases**

        **How it works:**
        1. Upload your documents in the sidebar
        2. Click Ingest Documents
        3. Ask questions in the chat
        4. Get cited answers with source attribution
        """)
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.caption(f"• {source}")

if prompt := st.chat_input("Ask anything about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                r = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "question": prompt,
                        "session_id": st.session_state.session_id
                    },
                    timeout=60
                )
                if r.status_code == 200:
                    result = r.json()
                    answer = result["answer"]
                    sources = result["sources"]
                else:
                    answer = f"Error: {r.json().get('detail', r.text)}"
                    sources = []
            except Exception as e:
                answer = f"Connection error: {e}"
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander("📚 Sources"):
                for source in sources:
                    st.caption(f"• {source}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
