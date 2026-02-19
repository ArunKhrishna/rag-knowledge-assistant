
import os
import sqlite3
import logging
import shutil
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    JSONLoader, UnstructuredXMLLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_community.chat_message_histories import ChatMessageHistory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================
# DOCUMENT LOADERS
# ==========================================

class SQLiteLoader:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def load(self) -> List[Document]:
        docs = []
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            for table in tables:
                df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                for idx, row in df.iterrows():
                    row_text = f"Table: {table}\n"
                    row_text += "\n".join([f"{col}: {val}" for col, val in row.items()])
                    docs.append(Document(
                        page_content=row_text,
                        metadata={
                            "source_type": "sqlite",
                            "file_path": self.db_path,
                            "file_name": Path(self.db_path).name,
                            "table_name": table,
                            "row_index": idx,
                            "loaded_at": datetime.now().isoformat()
                        }
                    ))
            conn.close()
        except Exception as e:
            logger.error(f"SQLite load error: {e}")
        return docs


class UnifiedDocumentLoader:
    SUPPORTED = ["pdf", "txt", "md", "csv", "json", "xml", "docx", "pptx", "db", "sqlite"]

    def __init__(self):
        self.loaded_docs: List[Document] = []
        self.load_stats: Dict[str, int] = {}

    def _ext(self, path: str) -> str:
        return Path(path).suffix.lower().replace(".", "")

    def _add_meta(self, docs, source_type, file_path):
        for doc in docs:
            doc.metadata.update({
                "source_type": source_type,
                "file_path": file_path,
                "file_name": Path(file_path).name,
                "loaded_at": datetime.now().isoformat()
            })
        return docs

    def load_file(self, file_path: str) -> List[Document]:
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return []

        ext = self._ext(file_path)
        docs = []

        try:
            if ext == "pdf":
                docs = PyPDFLoader(file_path).load()
                docs = self._add_meta(docs, "pdf", file_path)
            elif ext in ("txt", "md"):
                for enc in ["utf-8", "latin-1"]:
                    try:
                        docs = TextLoader(file_path, encoding=enc).load()
                        docs = self._add_meta(docs, "text", file_path)
                        break
                    except UnicodeDecodeError:
                        continue
            elif ext == "csv":
                docs = CSVLoader(file_path=file_path, encoding="utf-8").load()
                docs = self._add_meta(docs, "csv", file_path)
            elif ext == "json":
                docs = JSONLoader(file_path=file_path, jq_schema=".[]", text_content=False).load()
                docs = self._add_meta(docs, "json", file_path)
            elif ext == "xml":
                docs = UnstructuredXMLLoader(file_path).load()
                docs = self._add_meta(docs, "xml", file_path)
            elif ext == "docx":
                docs = UnstructuredWordDocumentLoader(file_path).load()
                docs = self._add_meta(docs, "docx", file_path)
            elif ext == "pptx":
                docs = UnstructuredPowerPointLoader(file_path).load()
                docs = self._add_meta(docs, "pptx", file_path)
            elif ext in ("db", "sqlite"):
                docs = SQLiteLoader(file_path).load()
            else:
                logger.warning(f"Unsupported format: {ext}")
                return []
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return []

        self.load_stats[ext] = self.load_stats.get(ext, 0) + len(docs)
        self.loaded_docs.extend(docs)
        return docs

    def load_directory(self, directory: str) -> List[Document]:
        all_docs = []
        for f in Path(directory).rglob("*"):
            if f.is_file() and self._ext(str(f)) in self.SUPPORTED:
                all_docs.extend(self.load_file(str(f)))
        return all_docs


# ==========================================
# CHUNKER
# ==========================================

class DocumentChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk(self, docs: List[Document]) -> List[Document]:
        chunks = self.splitter.split_documents(docs)
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({"chunk_id": i, "total_chunks": len(chunks)})
        return chunks


# ==========================================
# EMBEDDING MANAGER
# ==========================================

class EmbeddingManager:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        test = self.embeddings.embed_query("test")
        self.embedding_dimension = len(test)
        logger.info(f"Embedding model loaded: {model_name}, dim={self.embedding_dimension}")


# ==========================================
# VECTOR STORE MANAGER
# ==========================================

class VectorStoreManager:
    def __init__(self, embedding_manager: EmbeddingManager,
                 collection_name: str = "rag_knowledge_base",
                 location: str = ":memory:"):
        self.embedding_manager = embedding_manager
        self.collection_name = collection_name
        self.location = location
        self.client = QdrantClient(location=location)
        self.vectorstore = None

    def create_vectorstore(self, documents: List[Document]) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in existing:
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_manager.embedding_dimension,
                distance=Distance.COSINE
            )
        )
        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_manager.embeddings
        )
        self.vectorstore.add_documents(documents)
        count = self.client.get_collection(self.collection_name).points_count
        logger.info(f"Vector store created with {count} vectors")

    def add_documents(self, documents: List[Document]) -> None:
        if self.vectorstore:
            self.vectorstore.add_documents(documents)

    def get_retriever(self, k: int = 3, fetch_k: int = 10, lambda_mult: float = 0.6):
        if not self.vectorstore:
            return None
        return self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult}
        )

    def get_total_vectors(self) -> int:
        try:
            return self.client.get_collection(self.collection_name).points_count
        except Exception:
            return 0


# ==========================================
# CONVERSATION MEMORY
# ==========================================

class ConversationMemoryManager:
    def __init__(self, max_history_length: int = 10):
        self.max_history_length = max_history_length
        self.sessions: Dict[str, ChatMessageHistory] = {}
        self.session_metadata: Dict[str, Dict] = {}

    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatMessageHistory()
            self.session_metadata[session_id] = {
                "created_at": datetime.now().isoformat(),
            }
        return self.sessions[session_id]

    def clear_session(self, session_id: str) -> None:
        if session_id in self.sessions:
            self.sessions[session_id].clear()

    def get_session_stats(self, session_id: str) -> Dict:
        if session_id not in self.sessions:
            return {}
        history = self.sessions[session_id]
        messages = history.messages
        return {
            "total_messages": len(messages),
            "turns": len(messages) // 2,
            "created_at": self.session_metadata[session_id]["created_at"]
        }

    def trim_history(self, session_id: str) -> None:
        if session_id not in self.sessions:
            return
        history = self.sessions[session_id]
        max_msgs = self.max_history_length * 2
        if len(history.messages) > max_msgs:
            history.messages = history.messages[-max_msgs:]


# ==========================================
# RAG PIPELINE
# ==========================================

def format_context(docs: List[Document]) -> str:
    if not docs:
        return "No relevant context found."
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("file_name", "Unknown")
        parts.append(f"[Source {i}: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


SYSTEM_PROMPT = """You are an intelligent knowledge assistant with access to a curated knowledge base.

Guidelines:
- Answer based strictly on the provided context
- If the answer is not in the context, say: "I don't have enough information to answer this."
- Be concise, accurate, and structured
- Cite the source document name when referencing information
- Use conversation history to provide coherent multi-turn responses
- Do not fabricate information not present in the context"""

CONVERSATIONAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Context from knowledge base:\n{context}\n\nQuestion: {question}")
])


class RAGPipeline:
    """
    Unified RAG pipeline: document ingestion to conversational answering.
    """

    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key

        # Initialize components
        logger.info("Initializing RAG pipeline components...")
        self.loader = UnifiedDocumentLoader()
        self.chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
        self.embedding_manager = EmbeddingManager()
        self.vectorstore_manager = VectorStoreManager(
            embedding_manager=self.embedding_manager,
            collection_name="rag_knowledge_base",
            location=":memory:"
        )
        self.memory_manager = ConversationMemoryManager(max_history_length=10)

        # LLM
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            max_tokens=1024,
            groq_api_key=groq_api_key
        )
        self.output_parser = StrOutputParser()

        # Chain
        self.retriever = None
        self.chain = None
        self.is_ready = False

        logger.info("RAG pipeline initialized successfully")

    def ingest_directory(self, directory: str) -> Dict:
        """Load, chunk, and index all documents from a directory."""
        logger.info(f"Ingesting documents from: {directory}")
        docs = self.loader.load_directory(directory)
        if not docs:
            return {"status": "error", "message": "No documents found", "doc_count": 0}

        chunks = self.chunker.chunk(docs)
        self.vectorstore_manager.create_vectorstore(chunks)
        self._build_chain()

        return {
            "status": "success",
            "doc_count": len(docs),
            "chunk_count": len(chunks),
            "vector_count": self.vectorstore_manager.get_total_vectors(),
            "file_types": self.loader.load_stats
        }

    def ingest_file(self, file_path: str) -> Dict:
        """Load, chunk, and add a single file to the vector store."""
        logger.info(f"Ingesting file: {file_path}")
        docs = self.loader.load_file(file_path)
        if not docs:
            return {"status": "error", "message": f"Could not load {file_path}"}

        chunks = self.chunker.chunk(docs)

        if not self.is_ready:
            self.vectorstore_manager.create_vectorstore(chunks)
            self._build_chain()
        else:
            self.vectorstore_manager.add_documents(chunks)

        return {
            "status": "success",
            "doc_count": len(docs),
            "chunk_count": len(chunks),
            "vector_count": self.vectorstore_manager.get_total_vectors()
        }

    def _build_chain(self) -> None:
        """Build the conversational RAG chain."""
        self.retriever = self.vectorstore_manager.get_retriever(
            k=3, fetch_k=10, lambda_mult=0.6
        )
        self.chain = (
            {
                "context": RunnableLambda(
                    lambda x: format_context(self.retriever.invoke(x["question"]))
                ),
                "question": RunnableLambda(lambda x: x["question"]),
                "chat_history": RunnableLambda(lambda x: x.get("chat_history", []))
            }
            | CONVERSATIONAL_PROMPT
            | self.llm
            | self.output_parser
        )
        self.is_ready = True
        logger.info("RAG chain built and ready")

    def chat(self, question: str, session_id: str = "default") -> Dict:
        """Execute a conversational RAG query."""
        if not self.is_ready:
            return {
                "answer": "No documents ingested yet. Please upload documents first.",
                "sources": [],
                "session_id": session_id,
                "turn_number": 0
            }

        history = self.memory_manager.get_session_history(session_id)
        chat_history = history.messages

        retrieved_docs = self.retriever.invoke(question)
        answer = self.chain.invoke({
            "question": question,
            "chat_history": chat_history
        })

        history.add_user_message(question)
        history.add_ai_message(answer)
        self.memory_manager.trim_history(session_id)

        return {
            "answer": answer,
            "sources": list(set(doc.metadata.get("file_name", "Unknown") for doc in retrieved_docs)),
            "session_id": session_id,
            "turn_number": len(history.messages) // 2
        }

    def clear_session(self, session_id: str) -> None:
        self.memory_manager.clear_session(session_id)

    def get_stats(self) -> Dict:
        return {
            "is_ready": self.is_ready,
            "total_vectors": self.vectorstore_manager.get_total_vectors(),
            "file_types_loaded": self.loader.load_stats,
            "embedding_model": self.embedding_manager.model_name,
            "llm_model": "llama-3.3-70b-versatile"
        }
