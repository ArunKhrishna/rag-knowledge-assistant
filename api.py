
import os
import shutil
import tempfile
import logging
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# STARTUP: Initialize pipeline once
# ==========================================

pipeline: Optional[RAGPipeline] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_api_key:
        logger.warning("GROQ_API_KEY not set - pipeline not initialized")
    else:
        logger.info("Initializing RAG pipeline on startup...")
        pipeline = RAGPipeline(groq_api_key=groq_api_key)
        logger.info("RAG pipeline ready")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="RAG Knowledge Assistant API",
    description="Unified RAG system for natural language querying across all document formats",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# REQUEST / RESPONSE MODELS
# ==========================================

class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    session_id: str
    turn_number: int

class IngestResponse(BaseModel):
    status: str
    message: str
    doc_count: int = 0
    chunk_count: int = 0
    vector_count: int = 0
    file_types: dict = {}

class StatusResponse(BaseModel):
    status: str
    is_ready: bool
    total_vectors: int
    embedding_model: str
    llm_model: str


# ==========================================
# ENDPOINTS
# ==========================================

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "RAG Knowledge Assistant API"}


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "healthy",
        "pipeline_ready": pipeline is not None and pipeline.is_ready
    }


@app.get("/status", response_model=StatusResponse, tags=["Pipeline"])
def get_status():
    if pipeline is None:
        return StatusResponse(
            status="not_initialized",
            is_ready=False,
            total_vectors=0,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            llm_model="llama-3.3-70b-versatile"
        )
    stats = pipeline.get_stats()
    return StatusResponse(
        status="ready" if stats["is_ready"] else "awaiting_documents",
        is_ready=stats["is_ready"],
        total_vectors=stats["total_vectors"],
        embedding_model=stats["embedding_model"],
        llm_model=stats["llm_model"]
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Documents"])
async def ingest_documents(files: List[UploadFile] = File(...)):
    """Upload and ingest documents into the knowledge base."""
    global pipeline

    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized. Set GROQ_API_KEY environment variable."
        )

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    tmp_dir = tempfile.mkdtemp()
    try:
        for upload in files:
            file_path = os.path.join(tmp_dir, upload.filename)
            with open(file_path, "wb") as f:
                content = await upload.read()
                f.write(content)
            logger.info(f"Saved uploaded file: {upload.filename}")

        result = pipeline.ingest_directory(tmp_dir)

        if result["status"] == "error":
            raise HTTPException(status_code=422, detail=result["message"])

        return IngestResponse(
            status="success",
            message=f"Successfully ingested {result['doc_count']} documents",
            doc_count=result["doc_count"],
            chunk_count=result["chunk_count"],
            vector_count=result["vector_count"],
            file_types=result.get("file_types", {})
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest):
    """Send a question and get a RAG-powered answer."""
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized. Set GROQ_API_KEY environment variable."
        )

    if not pipeline.is_ready:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet. Call /ingest first."
        )

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = pipeline.chat(
        question=request.question,
        session_id=request.session_id
    )

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        session_id=result["session_id"],
        turn_number=result["turn_number"]
    )


@app.delete("/session/{session_id}", tags=["Chat"])
def clear_session(session_id: str):
    """Clear conversation history for a session."""
    if pipeline:
        pipeline.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}
