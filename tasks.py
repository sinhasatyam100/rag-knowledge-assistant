"""
tasks.py — Celery task definitions.

Runs in a separate worker process from the FastAPI server.
Handles the /ask endpoint (polling pattern).
/ask/stream bypasses Celery and runs inline in FastAPI.
"""

import os
import logging

from celery import Celery
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Celery setup ──────────────────────────────────────────────────────

REDIS_URL  = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "rag_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    result_expires=3600,
    worker_prefetch_multiplier=1,
)


# ── Worker startup initialisation ────────────────────────────────────
# These run once when the worker process starts — not on every task.

def _init_chroma_from_gcs():
    """Download ChromaDB index from GCS at worker startup."""
    bucket_name = os.getenv("GCS_BUCKET", "rag-knowledge-assistant-index")
    local_path  = "./chroma_db"
    try:
        from google.cloud import storage as gcs
        client = gcs.Client()
        bucket = client.bucket(bucket_name)
        blobs  = list(bucket.list_blobs(prefix="chroma_db/"))
        if blobs:
            os.makedirs(local_path, exist_ok=True)
            for blob in blobs:
                local_file = f"./{blob.name}"
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                blob.download_to_filename(local_file)
            logger.info(f"Worker: downloaded {len(blobs)} index files from GCS")
        else:
            logger.warning("Worker: no index found in GCS — starting empty")
    except Exception as e:
        logger.warning(f"Worker: GCS sync failed: {e}")


def _init_cross_encoder():
    """Pre-warm cross-encoder model at worker startup."""
    try:
        from retriever import get_cross_encoder
        get_cross_encoder()
        logger.info("Worker: cross-encoder pre-loaded")
    except Exception as e:
        logger.warning(f"Worker: could not pre-load cross-encoder: {e}")


_init_chroma_from_gcs()
_init_cross_encoder()


# ── RAG task ─────────────────────────────────────────────────────────

@celery_app.task(bind=True, max_retries=3)
def process_rag_query(self, question: str, top_k: int = 4) -> dict:
    """
    Runs the full RAG pipeline in a worker process.
    Called by /ask endpoint via .delay(). Result stored in Redis.
    Retries up to 3 times with exponential backoff on failure.
    """
    try:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from retriever import retrieve_and_rerank

        embeddings  = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings,
        )

        docs = retrieve_and_rerank(
            query=question,
            vectorstore=vectorstore,
            initial_k=min(top_k * 5, 20),
            final_k=top_k,
        )

        if not docs:
            return {
                "status":          "completed",
                "answer":          "No relevant documents found for your question.",
                "sources":         [],
                "question":        question,
                "retrieval_count": 0,
            }

        context = "\n\n".join([
            f"[Doc {i+1}]: {doc.page_content}" for i, doc in enumerate(docs)
        ])
        sources = [
            {
                "content":     doc.page_content[:200],
                "source":      doc.metadata.get("source", "unknown"),
                "chunk_index": i,
            }
            for i, doc in enumerate(docs)
        ]

        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("groq_api_key"),
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a helpful assistant.\n"
                "Use ONLY the following retrieved documents to answer the question.\n"
                "If the answer is not in the documents, say you don't know.\n"
                "Always cite which Doc number(s) you used.\n\n"
                "Retrieved documents:\n{context}"
            )),
            ("human", "{question}"),
        ])

        answer = (prompt | llm | StrOutputParser()).invoke({
            "context":  context,
            "question": question,
        })

        return {
            "status":          "completed",
            "answer":          answer,
            "sources":         sources,
            "question":        question,
            "retrieval_count": len(docs),
        }

    except Exception as exc:
        # Exponential backoff: 2s, 4s, 8s
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)
