# tasks.py

import os
from celery import Celery
import time

# ── Celery app setup ──────────────────────────────────────────────────

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "rag_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    result_expires=3600,
    worker_prefetch_multiplier=1,
)

# Module-level initialization — runs once when worker process starts
import os
from google.cloud import storage as gcs_client
import logging

logger = logging.getLogger(__name__)

def _init_chroma():
    """Download ChromaDB from GCS once at worker startup."""
    bucket_name = os.getenv("GCS_BUCKET", "rag-knowledge-assistant-index")
    local_path = "./chroma_db"
    
    try:
        client = gcs_client.Client()
        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix="chroma_db/"))
        if blobs:
            os.makedirs(local_path, exist_ok=True)
            for blob in blobs:
                local_file = f"./{blob.name}"
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                blob.download_to_filename(local_file)
            logger.info(f"Worker: Downloaded {len(blobs)} index files from GCS")
        else:
            logger.warning("Worker: No index found in GCS")
    except Exception as e:
        logger.warning(f"Worker: GCS sync failed: {e}")

# Run at module import time — once per worker process
_init_chroma()

# ── The RAG task ──────────────────────────────────────────────────────

@celery_app.task(bind=True, max_retries=3)
def process_rag_query(self, question: str, top_k: int = 4) -> dict:
    """
    This function runs in a WORKER PROCESS, not in your API.
    It receives the question, does the retrieval + LLM call,
    and returns the result. Celery stores the result in Redis
    automatically because we set a backend above.
    """
    try:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        # Load vector store
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )

        # Retrieve
        docs = vectorstore.similarity_search(question, k=top_k)

        if not docs:
            return {
                "status": "completed",
                "answer": "No relevant documents found.",
                "sources": [],
                "question": question,
                "retrieval_count": 0
            }

        # Format context
        context = "\n\n".join([
            f"[Doc {i+1}]: {doc.page_content}"
            for i, doc in enumerate(docs)
        ])

        sources = [
            {
                "content": doc.page_content[:200],
                "source": doc.metadata.get("source", "unknown"),
                "chunk_index": i
            }
            for i, doc in enumerate(docs)
        ]

        # LLM call
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("groq_api_key")
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant.
Use ONLY the following retrieved documents to answer the question.
If the answer is not in the documents, say you don't know.

Retrieved documents:
{context}"""),
            ("human", "{question}")
        ])

        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "context": context,
            "question": question
        })

        return {
            "status": "completed",
            "answer": answer,
            "sources": sources,
            "question": question,
            "retrieval_count": len(docs)
        }

    except Exception as exc:
        raise self.retry(
            exc=exc,
            countdown=2 ** self.request.retries
        )