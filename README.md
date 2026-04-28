# RAG Knowledge Assistant

A production-grade Retrieval-Augmented Generation API. Ask questions 
over indexed documents using semantic search and LLM inference.

## Live API
https://rag-api-308467052823.asia-south1.run.app/docs

## Architecture
- **FastAPI** — REST API with Pydantic validation, async streaming
- **ChromaDB** — vector store for semantic document retrieval (215 docs)
- **Celery + Redis** — async job queue for non-blocking LLM calls  
- **Groq (LLaMA 3.1 8B)** — fast LLM inference
- **sentence-transformers** — local CPU embeddings (all-MiniLM-L6-v2)

## GCP Infrastructure
- **Cloud Run** — API service (auto-scaling, HTTPS, asia-south1)
- **Cloud Run Worker Pool** — Celery workers (continuous background processing)
- **Cloud Memorystore** — managed Redis for job queue
- **Cloud Storage** — vector index persistence across container restarts
- **Secret Manager** — encrypted API key storage
- **Cloud Scheduler** — nightly reindex at 1am IST
- **VPC + VPC Connector** — private networking for Redis access

## Endpoints
| Endpoint | Method | Description |
|---|---|---|
| /health | GET | Service health + document count |
| /ask | POST | Async query → returns job_id |
| /status/{job_id} | GET | Poll for job result |
| /ask/stream | POST | Streaming response via SSE |
| /ingest | POST | Live document ingestion |
| /admin/reindex | POST | Triggered by Cloud Scheduler |

## Local Development
```bash
cp .env.example .env    # add your API keys
python ingest.py        # build vector index
docker-compose up       # starts api + worker + redis
```

## Evaluation
- Faithfulness, answer relevancy, context precision, 
  context recall via RAGAS (Month 4)
