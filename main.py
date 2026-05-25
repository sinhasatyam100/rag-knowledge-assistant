import os
import uuid
import shutil
import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse, StreamingResponse
from google.cloud import storage as gcs

from logs import logger
from exceptions import (
    RetrievalError, DocumentNotFoundError, LLMError,
    IngestionError, VectorStoreNotInitializedError,
)
from models import (
    QueryRequest,
    ConfluenceIngestRequest,
    JiraIngestRequest,
)
from retriever import get_cross_encoder, retrieve_and_rerank
from cache import cache_lookup, cache_store, cache_stats
from tasks import celery_app, process_rag_query


# ── GCS helpers ───────────────────────────────────────────────────────

def sync_chroma_from_gcs():
    """Download ChromaDB index from GCS at startup."""
    bucket_name = os.getenv("GCS_BUCKET", "rag-knowledge-assistant-index")
    local_path  = "./chroma_db"
    logger.info(f"Downloading ChromaDB index from gs://{bucket_name}/chroma_db/ ...")
    try:
        client = gcs.Client()
        bucket = client.bucket(bucket_name)
        blobs  = list(bucket.list_blobs(prefix="chroma_db/"))
        if not blobs:
            logger.warning("No index found in GCS. Starting with empty index.")
            return
        os.makedirs(local_path, exist_ok=True)
        for blob in blobs:
            local_file = f"./{blob.name}"
            os.makedirs(os.path.dirname(local_file), exist_ok=True)
            blob.download_to_filename(local_file)
        logger.info(f"Downloaded {len(blobs)} files from GCS.")
    except Exception as e:
        logger.warning(f"Could not download index from GCS: {e}. Starting with empty index.")


def sync_chroma_to_gcs(source_path: str = "./chroma_db"):
    """Upload ChromaDB index to GCS after ingestion."""
    bucket_name = os.getenv("GCS_BUCKET", "rag-knowledge-assistant-index")
    logger.info(f"Uploading {source_path} to GCS ...")
    try:
        client = gcs.Client()
        bucket = client.bucket(bucket_name)
        for root, dirs, files in os.walk(source_path):
            for file in files:
                local_file = os.path.join(root, file)
                gcs_path   = "chroma_db/" + os.path.relpath(local_file, source_path).replace("\\", "/")
                bucket.blob(gcs_path).upload_from_filename(local_file)
        logger.info("ChromaDB index uploaded to GCS.")
    except Exception as e:
        logger.error(f"Could not upload index to GCS: {e}")


# ── Lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG API ...")
    sync_chroma_from_gcs()
    get_cross_encoder()
    logger.info("Cross-encoder pre-warmed.")
    try:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings  = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings,
        )
        doc_count = vectorstore._collection.count()
        app.state.vectorstore = vectorstore
        app.state.embeddings  = embeddings
        logger.info(f"Vector store ready. Documents indexed: {doc_count}")
    except Exception as e:
        logger.critical(f"Failed to load vector store: {e}")
        raise

    yield

    logger.info("Shutting down RAG API ...")


app = FastAPI(
    title="RAG Knowledge Assistant",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Exception handlers ────────────────────────────────────────────────

@app.exception_handler(DocumentNotFoundError)
async def document_not_found_handler(request: Request, exc: DocumentNotFoundError):
    logger.warning(f"No documents found: {exc.message}")
    return JSONResponse(status_code=404, content={"error": exc.message})

@app.exception_handler(RetrievalError)
async def retrieval_error_handler(request: Request, exc: RetrievalError):
    logger.error(f"Retrieval error: {exc}")
    return JSONResponse(status_code=500, content={"error": exc.message})

@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError):
    logger.error(f"LLM error: {exc}")
    return JSONResponse(status_code=500, content={"error": exc.message})

@app.exception_handler(IngestionError)
async def ingestion_error_handler(request: Request, exc: IngestionError):
    logger.error(f"Ingestion error: {exc}")
    return JSONResponse(status_code=500, content={"error": exc.message})

@app.exception_handler(VectorStoreNotInitializedError)
async def vs_not_initialized_handler(request: Request, exc: VectorStoreNotInitializedError):
    logger.critical(f"Vector store not initialized: {exc}")
    return JSONResponse(status_code=503, content={"error": "Service unavailable, try again shortly"})


# ── Helper ────────────────────────────────────────────────────────────

def get_vectorstore(app_state):
    vs = getattr(app_state, "vectorstore", None)
    if vs is None:
        raise VectorStoreNotInitializedError("Vector store not loaded")
    return vs


# ── Routes ────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    try:
        vs        = get_vectorstore(app.state)
        doc_count = vs._collection.count()
        return {
            "status":             "healthy",
            "documents_indexed":  doc_count,
            "cache":              cache_stats(),
            "timestamp":          datetime.utcnow().isoformat(),
        }
    except VectorStoreNotInitializedError:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": "vector store not loaded"},
        )


@app.post("/ask")
def ask_question(request: QueryRequest):
    """
    Non-blocking query endpoint. Returns a job_id immediately.
    Poll /status/{job_id} for the result.
    """
    logger.info(f"Query received: '{request.question}' top_k={request.top_k}")
    task = process_rag_query.delay(request.question, request.top_k)
    logger.info(f"Task queued: {task.id}")
    return {
        "job_id":     task.id,
        "status":     "queued",
        "message":    "Query is being processed",
        "status_url": f"/status/{task.id}",
    }


@app.post("/ask/stream")
async def ask_stream(request: QueryRequest):
    """
    Streaming query endpoint. Returns SSE tokens in real time.
    Used by the Streamlit UI.
    """
    logger.info(f"Stream request: '{request.question}'")
    vs = get_vectorstore(app.state)

    # Embed query — used for both cache lookup and retrieval
    query_embedding = app.state.embeddings.embed_query(request.question)

    # Check semantic cache first
    cached = cache_lookup(query_embedding)
    if cached:
        async def stream_cached():
            yield f"data: {json.dumps({'type': 'cache_hit'})}\n\n"
            yield f"data: {json.dumps({'type': 'sources', 'sources': cached.get('sources', [])})}\n\n"
            words = cached["answer"].split(" ")
            for i, word in enumerate(words):
                token = word if i == len(words) - 1 else word + " "
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                await asyncio.sleep(0.01)
            yield f"data: {json.dumps({'type': 'done', 'full_response': cached['answer'], 'from_cache': True})}\n\n"

        return StreamingResponse(
            stream_cached(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    # Cache miss — run full RAG pipeline
    try:
        docs = retrieve_and_rerank(
            query=request.question,
            vectorstore=vs,
            initial_k=min(request.top_k * 5, 20),
            final_k=request.top_k,
        )
    except Exception as e:
        raise RetrievalError("Retrieval failed", original_error=e)

    if not docs:
        raise DocumentNotFoundError(f"No results found for: '{request.question}'")

    context = "\n\n".join([
        f"[Doc {i+1}]: {doc.page_content}" for i, doc in enumerate(docs)
    ])
    sources = [
        {"source": doc.metadata.get("source", "unknown"), "preview": doc.page_content[:150]}
        for doc in docs
    ]

    async def generate():
        try:
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            await asyncio.sleep(0.01)

            from langchain_groq import ChatGroq
            from langchain_core.prompts import ChatPromptTemplate

            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                api_key=os.getenv("groq_api_key"),
                streaming=True,
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

            full_response = ""
            async for chunk in (prompt | llm).astream(
                {"context": context, "question": request.question}
            ):
                token = chunk.content
                if token:
                    full_response += token
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            cache_store(
                query=request.question,
                query_embedding=query_embedding,
                result={"answer": full_response, "sources": sources, "retrieval_count": len(docs)},
            )
            yield f"data: {json.dumps({'type': 'done', 'full_response': full_response, 'from_cache': False})}\n\n"

        except Exception as e:
            logger.error(f"Stream generation error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "Connection":       "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/status/{job_id}")
def get_job_status(job_id: str):
    """Poll for the result of an /ask job."""
    from celery.result import AsyncResult
    task_result = AsyncResult(job_id, app=celery_app)

    state_map = {
        "PENDING": "queued",
        "STARTED": "processing",
        "RETRY":   "retrying",
    }
    if task_result.state in state_map:
        return {"job_id": job_id, "status": state_map[task_result.state]}
    if task_result.state == "SUCCESS":
        return {"job_id": job_id, "status": "completed", "result": task_result.result}
    if task_result.state == "FAILURE":
        return {"job_id": job_id, "status": "failed", "error": str(task_result.result)}
    return {"job_id": job_id, "status": task_result.state.lower()}


@app.post("/ingest")
async def ingest_documents(files: list[UploadFile] = File(...)):
    """
    Upload .txt or .pdf files into the knowledge base.
    Existing documents are preserved. GCS is updated once at the end.
    """
    logger.info(f"Ingest request: {len(files)} file(s)")
    vs         = get_vectorstore(app.state)
    embeddings = getattr(app.state, "embeddings", None)
    if embeddings is None:
        raise VectorStoreNotInitializedError("Embeddings not loaded")

    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)

    saved_paths = []
    results     = []

    # Save files to disk
    for file in files:
        if not file.filename.endswith((".txt", ".pdf")):
            results.append({"file": file.filename, "status": "rejected",
                             "reason": "Only .txt and .pdf files are supported"})
            logger.warning(f"Rejected: {file.filename} — unsupported type")
            continue
        safe_name = f"{uuid.uuid4()}_{file.filename}"
        file_path = upload_dir / safe_name
        try:
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            saved_paths.append((file.filename, file_path))
            logger.info(f"Saved: {file.filename} → {file_path}")
        except Exception as e:
            raise IngestionError(f"Failed to save {file.filename}", original_error=e)

    if not saved_paths:
        raise IngestionError("No valid files to process")

    # Process all files, then sync GCS once
    try:
        from langchain_community.document_loaders import TextLoader, PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter    = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        total_chunks = 0

        for original_name, file_path in saved_paths:
            try:
                loader    = PyPDFLoader(str(file_path)) if str(file_path).endswith(".pdf") \
                            else TextLoader(str(file_path), encoding="utf-8")
                documents = loader.load()
                for doc in documents:
                    doc.metadata["source"] = original_name
                chunks    = splitter.split_documents(documents)
                logger.info(f"{original_name}: {len(documents)} pages → {len(chunks)} chunks")
                vs.add_documents(chunks)
                total_chunks += len(chunks)
                results.append({"file": original_name, "status": "success",
                                 "chunks_added": len(chunks)})
            except Exception as e:
                logger.error(f"Failed to process {original_name}: {e}")
                results.append({"file": original_name, "status": "failed", "reason": str(e)})

        # Single GCS sync after all files are processed
        sync_chroma_to_gcs()

    except Exception as e:
        raise IngestionError("Document processing failed", original_error=e)

    final_count = vs._collection.count()
    logger.info(f"Ingest complete. Added {total_chunks} chunks. Total indexed: {final_count}")
    return {
        "message":                "Ingestion complete",
        "files_processed":        len(saved_paths),
        "total_chunks_added":     total_chunks,
        "total_documents_indexed": final_count,
        "results":                results,
    }


@app.post("/ingest/confluence")
async def ingest_confluence(request: ConfluenceIngestRequest):
    """
    Ingest pages from a Confluence space into the knowledge base.
    Use mock=true for a built-in demo without a real Confluence instance.
    """
    logger.info(
        f"Confluence ingest: space='{request.space_key}' "
        f"max_pages={request.max_pages} mock={request.mock}"
    )
    vs         = get_vectorstore(app.state)
    embeddings = getattr(app.state, "embeddings", None)
    if embeddings is None:
        raise VectorStoreNotInitializedError("Embeddings not loaded")

    try:
        from connectors.confluence import ConfluenceConnector
        connector = (
            ConfluenceConnector.mock() if request.mock
            else ConfluenceConnector(
                base_url=request.base_url,
                username=request.username,
                api_token=request.api_token,
            )
        )
        documents = connector.fetch_space(
            space_key=request.space_key,
            max_pages=request.max_pages,
        )
    except Exception as e:
        logger.error(f"Confluence fetch failed: {e}")
        raise IngestionError(f"Failed to fetch from Confluence: {e}", original_error=e)

    if not documents:
        return {
            "message":                "No pages found in this space",
            "pages_fetched":          0,
            "chunks_added":           0,
            "total_documents_indexed": vs._collection.count(),
        }

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        logger.info(f"Confluence: {len(documents)} pages → {len(chunks)} chunks")
        vs.add_documents(chunks)
        sync_chroma_to_gcs()
    except Exception as e:
        logger.error(f"Confluence indexing failed: {e}")
        raise IngestionError("Failed to index Confluence documents", original_error=e)

    final_count = vs._collection.count()
    logger.info(f"Confluence ingest done. chunks={len(chunks)} total={final_count}")
    return {
        "message":                "Confluence ingestion complete",
        "space_key":              request.space_key,
        "pages_fetched":          len(documents),
        "chunks_added":           len(chunks),
        "total_documents_indexed": final_count,
        "mock":                   request.mock,
    }


@app.post("/ingest/jira")
async def ingest_jira(request: JiraIngestRequest):
    """
    Ingest issues from a JIRA project into the knowledge base.
    Use mock=true for a built-in demo without a real JIRA instance.
    """
    logger.info(
        f"JIRA ingest: project='{request.project_key}' "
        f"max_issues={request.max_issues} mock={request.mock}"
    )
    vs         = get_vectorstore(app.state)
    embeddings = getattr(app.state, "embeddings", None)
    if embeddings is None:
        raise VectorStoreNotInitializedError("Embeddings not loaded")

    try:
        from connectors.jira import JiraConnector
        connector = (
            JiraConnector.mock() if request.mock
            else JiraConnector(
                base_url=request.base_url,
                username=request.username,
                api_token=request.api_token,
            )
        )
        documents = connector.fetch_project(
            project_key=request.project_key,
            max_issues=request.max_issues,
        )
    except Exception as e:
        logger.error(f"JIRA fetch failed: {e}")
        raise IngestionError(f"Failed to fetch from JIRA: {e}", original_error=e)

    if not documents:
        return {
            "message":                "No issues found in this project",
            "issues_fetched":         0,
            "chunks_added":           0,
            "total_documents_indexed": vs._collection.count(),
        }

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        logger.info(f"JIRA: {len(documents)} issues → {len(chunks)} chunks")
        vs.add_documents(chunks)
        sync_chroma_to_gcs()
    except Exception as e:
        logger.error(f"JIRA indexing failed: {e}")
        raise IngestionError("Failed to index JIRA documents", original_error=e)

    final_count = vs._collection.count()
    logger.info(f"JIRA ingest done. chunks={len(chunks)} total={final_count}")
    return {
        "message":                "JIRA ingestion complete",
        "project_key":            request.project_key,
        "issues_fetched":         len(documents),
        "chunks_added":           len(chunks),
        "total_documents_indexed": final_count,
        "mock":                   request.mock,
    }


@app.post("/admin/reindex")
async def trigger_reindex():
    """
    Reload the ChromaDB index from GCS and refresh app.state.
    Use this after uploading a fresh index to GCS externally
    (e.g. after running ingest.py --upload locally).
    Triggered by Cloud Scheduler or manually.
    """
    logger.info("Reindex triggered — reloading from GCS")
    try:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings

        # Re-download the latest index from GCS
        sync_chroma_from_gcs()

        # Reload the vectorstore into app.state
        embeddings  = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings,
        )
        app.state.vectorstore = vectorstore
        app.state.embeddings  = embeddings

        count = vectorstore._collection.count()
        logger.info(f"Reindex complete. {count} chunks now live.")
        return {"status": "success", "chunks_indexed": count}

    except Exception as e:
        logger.error(f"Reindex failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
