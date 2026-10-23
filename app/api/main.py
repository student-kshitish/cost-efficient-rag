import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.embeddings.embedder import Embedder
from app.storage.vector_store import VectorStore
from app.retrieval.search import RetrievalService
from app.generation.answer import AnswerService


# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# API request and response models
# --------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10000)
    top_k: int = Field(default=settings.top_k, ge=1, le=20)
    document_type: str | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[dict]
    retrieved_chunks: int
    latency_ms: float
    model: str
    usage: dict


# --------------------------------------------------
# Application lifecycle
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing RAG application")

    embedder = Embedder(model_name=settings.embedding_model)

    store = VectorStore(
        db_path=settings.vector_db_path,
        dimension=embedder.dimension,
    )

    retriever = RetrievalService(
        embedder=embedder,
        store=store,
        top_k=settings.top_k,
    )

    answer_service = AnswerService(
        retriever=retriever,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
    )

    app.state.embedder = embedder
    app.state.store = store
    app.state.retriever = retriever
    app.state.answer_service = answer_service

    logger.info("RAG application initialized")
    yield
    logger.info("Shutting down RAG application")


app = FastAPI(
    title=settings.app_name,
    description="A cost-efficient retrieval-augmented generation API",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "app": settings.app_name,
        "environment": settings.app_env,
    }


# --------------------------------------------------
# Ask endpoint
# --------------------------------------------------

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty",
        )

    started = time.perf_counter()

    try:
        result = app.state.answer_service.answer_question(
            question=question,
            top_k=request.top_k,
            document_type=request.document_type,
        )

        elapsed_ms = round(
            (time.perf_counter() - started) * 1000,
            2,
        )

        logger.info(
            "RAG request completed | latency_ms=%s | chunks=%s | model=%s",
            elapsed_ms,
            result.get("retrieved_chunks", 0),
            result.get("model"),
        )

        result["latency_ms"] = elapsed_ms
        return result

    except Exception as exc:
        logger.exception("RAG request failed")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate an answer. Check server logs.",
        ) from exc


# --------------------------------------------------
# Stats endpoint
# --------------------------------------------------

@app.get("/stats")
def stats():
    try:
        chunk_count = app.state.store.count()

        return {
            "indexed_chunks": chunk_count,
            "embedding_model": settings.embedding_model,
            "llm_model": settings.llm_model,
            "default_top_k": settings.top_k,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        }

    except Exception as exc:
        logger.exception("Could not retrieve index statistics")
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve index statistics.",
        ) from exc
