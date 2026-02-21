# Unified RAG-Based Intelligent Knowledge Assistant

A production-grade Retrieval-Augmented Generation (RAG) system that enables natural language, context-aware querying across multiple document formats through a single conversational interface.

---

## Problem Statement

Organizations store knowledge across documents, datasets, and databases, making information retrieval fragmented and inefficient. This system provides a unified interface for natural language querying across all knowledge formats:

- **Unstructured:** PDF, Word (`.docx`), PowerPoint (`.pptx`), TXT, Markdown
- **Semi-structured:** JSON, XML
- **Structured:** CSV datasets, SQLite databases

---

## System Architecture

```
User Query
→ Streamlit UI (frontend)
→ FastAPI Backend (REST API)
→ UnifiedDocumentLoader  (Phase 1.1)
→ DocumentChunker        (Phase 1.2)
→ EmbeddingManager       (Phase 1.3)
→ VectorStoreManager     (Phase 1.4)
→ MMRRetriever           (Phase 2.1)
→ ConversationalRAGChain (Phase 3.1 + 3.2)
→ ConversationMemoryManager (Phase 4)
→ Answer + Source Citations
```

---

## Tech Stack

| Layer | Technology | Details |
|---|---|---|
| LLM | Groq | `llama-3.3-70b-versatile`, temp=0.0 |
| Embeddings | HuggingFace | `sentence-transformers/all-MiniLM-L6-v2`, 384-dim |
| Vector DB | Qdrant | In-memory, Cosine similarity |
| RAG Framework | LangChain LCEL | Full expression language pipeline |
| Frontend | Streamlit | File upload, chat UI, source citations |
| Backend API | FastAPI + Uvicorn | REST endpoints, CORS, Pydantic models |
| Containerization | Docker Compose | Separate backend + frontend containers |

---

## Project Structure

```
rag_app/
├── api.py                  # FastAPI backend (REST API + pipeline init)
├── app.py                  # Streamlit frontend (calls FastAPI)
├── core/
│   └── pipeline.py         # Complete RAG backend (all components)
├── requirements.txt        # All Python dependencies
├── Dockerfile.backend      # Backend container (pre-downloads embedding model)
├── Dockerfile.frontend     # Frontend container (streamlit + requests only)
├── docker-compose.yml      # Orchestration with health checks
├── .env.example            # Environment variable template
├── .gitignore
└── README.md
```

---

## Quick Start

### Docker (Recommended)

```bash
# 1. Clone the repo
git clone https://github.com/your_username/rag-knowledge-assistant
cd rag-knowledge-assistant

# 2. Set your Groq API key
echo "GROQ_API_KEY=gsk_your_key_here" > .env

# 3. Build and start
docker compose up --build

# 4. Open the app
# Frontend: http://localhost:8501
# API docs: http://localhost:8000/docs
```

### Google Colab

1. Open the notebook in Colab
2. Run all cells top to bottom
3. Set `GROQ_API_KEY` in Colab Secrets (Notebook access: **ON**)
4. Upload documents via sidebar
5. Start querying

---

## Phase 1: Document Ingestion Layer

### Stage 1.1 — Multi-Format Document Loaders

A `UnifiedDocumentLoader` class handles all 9 supported formats via automatic format detection from file extension.

**Unstructured Documents:**
- PDF → `PyPDFLoader` (one Document per page)
- Word `.docx` → `UnstructuredWordDocumentLoader`
- PowerPoint `.pptx` → `UnstructuredPowerPointLoader`
- Plain text / Markdown `.txt`, `.md` → `TextLoader` with UTF-8 → latin-1 fallback

**Semi-Structured Data:**
- JSON → `JSONLoader` with jq schema `.[]`
- XML → `UnstructuredXMLLoader`

**Structured Data:**
- CSV → `CSVLoader` (one Document per row)
- SQLite `.db`, `.sqlite` → custom `SQLiteLoader` (reads all tables; each row becomes a Document with table name + column metadata)

All loaded documents receive standardized metadata:

```python
{
    "source_type": "pdf" | "text" | "csv" | "json" | "xml" | "docx" | "pptx" | "sqlite",
    "file_path":   "/path/to/file",
    "file_name":   "filename.ext",
    "loaded_at":   "2026-02-21T00:00:00"
}
```

### Stage 1.2 — Document Chunking Strategy

`DocumentChunker` wraps LangChain's `RecursiveCharacterTextSplitter`:

| Parameter | Value | Rationale |
|---|---|---|
| `chunk_size` | 1000 characters | Fits within embedding model context |
| `chunk_overlap` | 200 characters | 20% overlap preserves cross-chunk context |
| `separators` | `["\n\n", "\n", ". ", " ", ""]` | Paragraph → sentence → word priority |
| `length_function` | `len` | Character-based, not token-based |

Each chunk preserves all parent document metadata plus:

```python
{"chunk_id": 0, "total_chunks": 12}
```

### Stage 1.3 — Embedding Model

`EmbeddingManager` wraps HuggingFace sentence-transformers:

| Setting | Value |
|---|---|
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Dimensions | 384 |
| Device | CPU |
| Normalization | L2 normalized (cosine-ready) |
| API key needed | No |

### Stage 1.4 — Vector Database

`VectorStoreManager` uses Qdrant (chosen over ChromaDB for zero dependency conflicts in production):

| Setting | Value |
|---|---|
| Storage | In-memory (`:memory:`) |
| Distance metric | Cosine similarity |
| Collection name | `rag_knowledge_base` |
| Re-index on new docs | `add_documents()` appends incrementally |

> **Note:** In-memory mode means vectors reset on container restart. For persistent deployment, switch `location=":memory:"` to `location="./qdrant_storage"` in `VectorStoreManager.__init__`.

---

## Phase 2: Query Processing & Retrieval

### Stage 2.1 — MMR Retrieval

Maximal Marginal Relevance (MMR) is used instead of plain similarity search to balance two objectives simultaneously:

- **Relevance:** Retrieved chunks are similar to the query
- **Diversity:** Retrieved chunks are dissimilar to each other

This prevents redundant context (e.g., 3 chunks saying the same thing) from being sent to the LLM, improving answer quality.

| Parameter | Value | Effect |
|---|---|---|
| `k` | 3 | Final documents returned |
| `fetch_k` | 10 | Candidate pool before MMR reranking |
| `lambda_mult` | 0.6 | 0 = pure diversity, 1 = pure relevance |

### Stage 2.2 — Retrieval Configuration

MMR is configured via `vectorstore.as_retriever(search_type="mmr")`. Every retrieved chunk carries `file_name` metadata for source attribution, and context is formatted as:

```
[Source 1: machine_learning.txt]
<chunk content>

---

[Source 2: models.csv]
<chunk content>
```

---

## Phase 3: Generation Layer

### Stage 3.1 — LLM Integration

Groq is used for fast inference with `llama-3.3-70b-versatile`:

| Setting | Value |
|---|---|
| Provider | Groq (`langchain-groq`) |
| Model | `llama-3.3-70b-versatile` |
| Temperature | 0.0 (deterministic) |
| Max tokens | 1024 |
| API key source | Environment variable `GROQ_API_KEY` |

The system prompt enforces strict grounding:
- Answer only from provided context
- Refuse with *"I don't have enough information"* if context is absent
- Cite source document names inline
- No hallucination or inference beyond context

### Stage 3.2 — RAG Chain (LCEL)

The full pipeline is assembled using LangChain Expression Language (LCEL):

```python
chain = (
    {
        "context":      RunnableLambda(lambda x: format_context(retriever.invoke(x["question"]))),
        "question":     RunnableLambda(lambda x: x["question"]),
        "chat_history": RunnableLambda(lambda x: x.get("chat_history", []))
    }
    | CONVERSATIONAL_PROMPT
    | llm
    | StrOutputParser()
)
```

---

## Phase 4: Conversational Memory

`ConversationMemoryManager` provides per-session isolated memory:

| Feature | Implementation |
|---|---|
| Session isolation | Each `session_id` gets its own `ChatMessageHistory` |
| Memory retention | Last 10 turns (20 messages) per session |
| Automatic trimming | Oldest messages dropped when limit exceeded |
| Session clearing | `DELETE /session/{session_id}` API endpoint |
| Multi-turn awareness | Full chat history passed to LLM on every turn |

This enables the model to correctly handle follow-up questions like *"What are the main types you mentioned?"* without re-explaining prior context.

---

## Phase 5: Deployment

### Stage 5.1 — FastAPI Backend (`api.py`)

REST API exposing the RAG pipeline:

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check |
| `/health` | GET | Pipeline readiness status |
| `/status` | GET | Vector count, model info |
| `/ingest` | POST | Upload + index documents (multipart) |
| `/chat` | POST | Send question, get answer + sources |
| `/session/{id}` | DELETE | Clear conversation history |

The pipeline initializes at startup via a `lifespan` context manager using `GROQ_API_KEY` from environment — no runtime key entry needed.

### Stage 5.2 — Streamlit Frontend (`app.py`)

- Calls FastAPI backend via `requests` — fully decoupled from the pipeline
- `API_URL` configurable via environment variable (default: `http://localhost:8000`)
- Features: file uploader, document ingestion button, knowledge base metrics, chat interface with source citation expanders, session clear button

### Stage 5.3 — Docker Compose Deployment

Two separate containers with health-check dependency:

```yaml
services:
  backend:   # FastAPI + RAG pipeline, port 8000
  frontend:  # Streamlit UI, port 8501
             # starts only after backend is healthy
```

---

## Supported Document Formats

| Format | Extension | Loader | Notes |
|---|---|---|---|
| PDF | `.pdf` | `PyPDFLoader` | One doc per page |
| Word | `.docx` | `UnstructuredWordDocumentLoader` | Full text extraction |
| PowerPoint | `.pptx` | `UnstructuredPowerPointLoader` | Slide text extraction |
| Text | `.txt` | `TextLoader` | UTF-8 + latin-1 fallback |
| Markdown | `.md` | `TextLoader` | Same as text |
| CSV | `.csv` | `CSVLoader` | One doc per row |
| JSON | `.json` | `JSONLoader` | jq schema `.[]` |
| XML | `.xml` | `UnstructuredXMLLoader` | Full element extraction |
| SQLite | `.db`, `.sqlite` | `SQLiteLoader` (custom) | All tables, one doc per row |

---

## Production-Grade Features

| Feature | Implementation |
|---|---|
| Error handling | Try/except at every loader, encoding fallback, graceful degradation |
| Logging | `logging` module throughout all classes with INFO/ERROR levels |
| Hallucination guard | System prompt enforces context-only answers |
| API validation | Pydantic request/response models on all FastAPI endpoints |
| CORS | Enabled on FastAPI for cross-origin frontend access |
| Health checks | Docker `HEALTHCHECK` + `/health` endpoint |
| Session isolation | Per-session memory prevents cross-user contamination |
| Stateless API | FastAPI backend is fully stateless per request |
| Env-based secrets | No hardcoded API keys anywhere |
| Layer caching | Dockerfile copies `requirements.txt` before source for fast rebuilds |
| Model pre-warming | Embedding model downloaded at Docker build time, not runtime |
