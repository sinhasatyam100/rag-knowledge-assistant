# main.py
import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
# os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
groq_api_key = os.getenv("groq_api_key")

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from models import QueryRequest, QueryResponse, SourceDocument
from exceptions import (
    RetrievalError, DocumentNotFoundError, LLMError,
    IngestionError, VectorStoreNotInitializedError, RAGException
)
from logs import logger
from datetime import datetime
from pathlib import Path
import time
import shutil
import uuid
from tasks import celery_app, process_rag_query
from fastapi.responses import StreamingResponse
import asyncio
from google.cloud import storage as gcs
import shutil
import os
from requests import Request

def sync_chroma_from_gcs():
    """Download ChromaDB index from GCS using Python client."""
    bucket_name = os.getenv("GCS_BUCKET", "rag-knowledge-assistant-index")
    local_path = "./chroma_db"
    
    logger.info(f"Downloading ChromaDB index from gs://{bucket_name}/chroma_db/...")
    try:
        client = gcs.Client()
        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix="chroma_db/"))
        
        if not blobs:
            logger.warning("No index found in GCS. Starting with empty index.")
            return
            
        os.makedirs(local_path, exist_ok=True)
        
        for blob in blobs:
            # blob.name = "chroma_db/chroma.sqlite3" etc
            local_file = f"./{blob.name}"
            os.makedirs(os.path.dirname(local_file), exist_ok=True)
            blob.download_to_filename(local_file)
        
        logger.info(f"Downloaded {len(blobs)} files from GCS")
        
    except Exception as e:
        logger.warning(f"Could not download index from GCS: {e}. Starting with empty index.")

def sync_chroma_to_gcs(source_path="./chroma_db"):
    bucket_name = os.getenv("GCS_BUCKET", "rag-knowledge-assistant-index")
    logger.info(f"Uploading {source_path} to GCS...")
    try:
        client = gcs.Client()
        bucket = client.bucket(bucket_name)
        for root, dirs, files in os.walk(source_path):
            for file in files:
                local_file = os.path.join(root, file)
                gcs_path = "chroma_db/" + os.path.relpath(local_file, source_path).replace("\\", "/")
                blob = bucket.blob(gcs_path)
                blob.upload_from_filename(local_file)
        logger.info("ChromaDB index uploaded to GCS successfully")
    except Exception as e:
        logger.error(f"Could not upload index to GCS: {e}")
        
        # ── Lifespan ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG API...")
    sync_chroma_from_gcs()

    try:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        print("Loading vector store and embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("Initializing Chroma vector store...")
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
        doc_count = vectorstore._collection.count()
        print(f"Vector store loaded with {doc_count} documents")
        # Store BOTH on app.state — we need embeddings in /ingest too
        app.state.vectorstore = vectorstore
        app.state.embeddings = embeddings

        logger.info(f"Vector store loaded. Documents indexed: {doc_count}")

    except Exception as e:
        # CRITICAL because the app is useless without the vector store
        logger.critical(f"Failed to load vector store on startup: {e}")
        raise  # Re-raise — don't silently start a broken server

    yield

    logger.info("Shutting down RAG API...")


app = FastAPI(
    title="RAG Knowledge Assistant",
    version="1.0.0",
    lifespan=lifespan
)


# ── Exception handlers ───────────────────────────────────────────────

@app.exception_handler(DocumentNotFoundError)
async def document_not_found_handler(request, exc: DocumentNotFoundError):
    logger.warning(f"No documents found: {exc.message}")
    return JSONResponse(status_code=404, content={"error": exc.message})

@app.exception_handler(RetrievalError)
async def retrieval_error_handler(request, exc: RetrievalError):
    logger.error(f"Retrieval error: {exc}")
    return JSONResponse(status_code=500, content={"error": exc.message})

@app.exception_handler(LLMError)
async def llm_error_handler(request, exc: LLMError):
    logger.error(f"LLM error: {exc}")
    return JSONResponse(status_code=500, content={"error": exc.message})

@app.exception_handler(IngestionError)
async def ingestion_error_handler(request, exc: IngestionError):
    logger.error(f"Ingestion error: {exc}")
    return JSONResponse(status_code=500, content={"error": exc.message})

@app.exception_handler(VectorStoreNotInitializedError)
async def vector_store_not_initialized_handler(request, exc):
    logger.critical(f"Vector store not initialized: {exc}")
    return JSONResponse(status_code=503, content={"error": "Service unavailable, try again shortly"})


# ── Helper ────────────────────────────────────────────────────────────
def get_vectorstore(app_state):
    """
    Safely retrieves vectorstore from app.state.
    Raises VectorStoreNotInitializedError if missing.
    Called at the start of any route that needs the vector store.
    """
    vs = getattr(app_state, "vectorstore", None)
    if vs is None:
        raise VectorStoreNotInitializedError("Vector store not loaded")
    return vs


# ── Routes ────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    try:
        vs = get_vectorstore(app.state)
        doc_count = vs._collection.count()
        return {
            "status": "healthy",
            "documents_indexed": doc_count,
            "timestamp": datetime.utcnow().isoformat()
        }
    except VectorStoreNotInitializedError:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": "vector store not loaded"}
        )

@app.post("/admin/reindex")
async def trigger_reindex():
    """
    Triggered by Cloud Scheduler nightly.
    Re-runs ingestion from all source URLs and uploads to GCS.
    Protected — only callable from Cloud Scheduler via OIDC token.
    """
    logger.info("Reindex triggered")
    
    try:
        from langchain_community.document_loaders import WebBaseLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        import bs4, shutil

        URLS = [
            "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "https://en.wikipedia.org/wiki/Machine_learning",
            "https://en.wikipedia.org/wiki/Natural_language_processing",
        ]

        # Load
        loader = WebBaseLoader(
            web_paths=URLS,
            bs_kwargs=dict(parse_only=bs4.SoupStrainer("p"))
        )
        documents = loader.load()

        # Chunk
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        logger.info(f"Reindex: {len(chunks)} chunks from {len(URLS)} URLs")

        # Rebuild index
        shutil.rmtree("/tmp/chroma_db", ignore_errors=True)

        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        
        vectorstore = Chroma.from_documents(
            chunks, embeddings,
            persist_directory="/tmp/chroma_db"  # /tmp is always writable
        )
        # Upload to GCS
        sync_chroma_to_gcs(source_path="/tmp/chroma_db")

        # And update app.state to point to new location
        # Update app.state so running instance uses new index
        app.state.vectorstore = vectorstore
        count = vectorstore._collection.count()
        logger.info(f"Reindex complete. {count} chunks indexed.")

        return {
            "status": "success",
            "chunks_indexed": count,
            "sources": len(URLS)
        }

    except Exception as e:
        logger.error(f"Reindex failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
def ask_question(request: QueryRequest):
    """
    Now returns immediately with a job_id.
    The actual RAG processing happens in a background worker.
    """
    logger.info(f"Query received: '{request.question}' top_k={request.top_k}")

    # .delay() puts the task on the Redis queue and returns immediately
    # It does NOT wait for the task to complete
    task = process_rag_query.delay(
        request.question,
        request.top_k
    )

    logger.info(f"Task queued: {task.id}")

    # Return immediately — milliseconds, not seconds
    return {
        "job_id": task.id,
        "status": "queued",
        "message": "Query is being processed",
        "status_url": f"/status/{task.id}"
    }

@app.post("/ask/stream")
async def ask_stream(request: QueryRequest):
    """
    Streaming version of /ask.
    Returns tokens as they're generated instead of waiting for completion.
    Uses Server-Sent Events (SSE) format.
    
    Unlike /ask which returns a job_id, this keeps the connection open
    and pushes tokens directly to the client as the LLM generates them.
    
    Use this for: chat UI, interactive interfaces, real-time feedback.
    Use /ask for: batch processing, async pipelines, background jobs.
    """
    logger.info(f"Stream request: '{request.question}'")

    # Get vector store
    vs = get_vectorstore(app.state)

    # ── Retrieval (same as /ask) ──────────────────────────────────────
    try:
        docs = vs.similarity_search(request.question, k=request.top_k)
    except Exception as e:
        raise RetrievalError("Similarity search failed", original_error=e)

    if not docs:
        raise DocumentNotFoundError(f"No results for: '{request.question}'")

    # Format context
    context = "\n\n".join([
        f"[Doc {i+1}]: {doc.page_content}"
        for i, doc in enumerate(docs)
    ])

    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "preview": doc.page_content[:150]
        }
        for doc in docs
    ]

    # ── Async generator — the heart of streaming ──────────────────────
    async def generate():
        try:
            import json
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

            await asyncio.sleep(0.01)

            # ── LLM streaming call ────────────────────────────────────
            from langchain_groq import ChatGroq
            from langchain_core.prompts import ChatPromptTemplate

            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                api_key=os.getenv("groq_api_key"),
                streaming=True
            )

            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant.
Use ONLY the following retrieved documents to answer the question.
If the answer is not in the documents, say you don't know.

Retrieved documents:
{context}"""),
                ("human", "{question}")
            ])

            full_response = ""
            async for chunk in (prompt | llm).astream({
                "context": context,
                "question": request.question
            }):
                token = chunk.content
                if token:
                    full_response += token
                    # Send each token as an SSE event
                    # type: 'token' tells the client this is answer text
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'full_response': full_response, 'retrieval_count': len(docs)})}\n\n"

            logger.info(f"Stream completed for: '{request.question}'")

        except Exception as e:
            logger.error(f"Stream error: {e}")
            import json
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/status/{job_id}")
def get_job_status(job_id: str):
    """
    Poll this endpoint to check if your job is done.
    
    Celery task states:
    PENDING  → task is queued, worker hasn't picked it up yet
    STARTED  → worker picked it up, currently processing
    SUCCESS  → done, result is available
    FAILURE  → something went wrong after all retries
    RETRY    → failed once, waiting to retry
    """
    from celery.result import AsyncResult

    task_result = AsyncResult(job_id, app=celery_app)

    if task_result.state == "PENDING":
        return {"job_id": job_id, "status": "queued"}

    elif task_result.state == "STARTED":
        return {"job_id": job_id, "status": "processing"}

    elif task_result.state == "SUCCESS":
        return {
            "job_id": job_id,
            "status": "completed",
            "result": task_result.result
        }

    elif task_result.state == "FAILURE":
        return {
            "job_id": job_id,
            "status": "failed",
            "error": str(task_result.result)
        }

    elif task_result.state == "RETRY":
        return {"job_id": job_id, "status": "retrying"}

    return {"job_id": job_id, "status": task_result.state.lower()}

@app.post("/ingest")
async def ingest_documents(files: list[UploadFile] = File(...)):
    """
    Upload one or more documents to be added to the knowledge base.
    Supports .txt and .pdf files.
    The server keeps running — existing documents are preserved.
    New chunks are immediately searchable after this returns.
    """
    logger.info(f"Ingest request received: {len(files)} file(s)")

    vs = get_vectorstore(app.state)
    embeddings = getattr(app.state, "embeddings", None)
    if embeddings is None:
        raise VectorStoreNotInitializedError("Embeddings not loaded")

    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)

    saved_paths = []
    results = []

    # ── Save uploaded files to disk ───────────────────────────────────
    for file in files:
        if not file.filename.endswith((".txt", ".pdf")):
            results.append({
                "file": file.filename,
                "status": "rejected",
                "reason": "Only .txt and .pdf files are supported"
            })
            logger.warning(f"Rejected file: {file.filename} — unsupported type")
            continue

        safe_filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = upload_dir / safe_filename

        try:
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            saved_paths.append((file.filename, file_path))
            logger.info(f"Saved: {file.filename} → {file_path}")
        except Exception as e:
            raise IngestionError(
                f"Failed to save file: {file.filename}",
                original_error=e
            )

    if not saved_paths:
        raise IngestionError("No valid files to process")

    # ── Load, chunk, embed, add to ChromaDB ───────────────────────────
    try:
        from langchain_community.document_loaders import TextLoader, PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        total_chunks = 0

        for original_name, file_path in saved_paths:
            try:
                # Choose loader based on file type
                if str(file_path).endswith(".pdf"):
                    loader = PyPDFLoader(str(file_path))
                else:
                    loader = TextLoader(str(file_path), encoding="utf-8")

                documents = loader.load()

                for doc in documents:
                    doc.metadata["source"] = original_name

                chunks = splitter.split_documents(documents)
                logger.info(f"{original_name}: {len(documents)} pages → {len(chunks)} chunks")

                vs.add_documents(chunks)
                sync_chroma_to_gcs()
                total_chunks += len(chunks)

                results.append({
                    "file": original_name,
                    "status": "success",
                    "chunks_added": len(chunks)
                })

            except Exception as e:
                logger.error(f"Failed to process {original_name}: {e}")
                results.append({
                    "file": original_name,
                    "status": "failed",
                    "reason": str(e)
                })

    except Exception as e:
        raise IngestionError("Document processing failed", original_error=e)

    final_count = vs._collection.count()
    logger.info(f"Ingest complete. Added {total_chunks} chunks. Total indexed: {final_count}")

    return {
        "message": f"Ingestion complete",
        "files_processed": len(saved_paths),
        "total_chunks_added": total_chunks,
        "total_documents_indexed": final_count,
        "results": results
    }