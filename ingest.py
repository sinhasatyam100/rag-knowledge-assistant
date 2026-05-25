"""
ingest.py — Initial knowledge base builder.

Loads documents from a local folder or explicit file list,
chunks them, embeds with MiniLM, and stores in ChromaDB.

Run once to bootstrap the index before starting the API:
    python ingest.py

Supported file types: .txt, .pdf

Place your documents in the ./documents/ folder and run this script.
The resulting ChromaDB index is saved to ./chroma_db/ and can be
uploaded to GCS with: python ingest.py --upload
"""

import os
import sys
import time
import argparse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

CHUNK_SIZE        = 1000
CHUNK_OVERLAP     = 200
PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
DOCUMENTS_DIR     = "./documents"


def load_documents(source_dir: str) -> list:
    """
    Load all .txt and .pdf files from a directory.
    Returns a list of LangChain Document objects.
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"Directory '{source_dir}' not found. Creating it.")
        source_path.mkdir(parents=True)
        print(f"Place your .txt and .pdf files in '{source_dir}' and re-run.")
        sys.exit(0)

    files = list(source_path.glob("*.txt")) + list(source_path.glob("*.pdf"))
    if not files:
        print(f"No .txt or .pdf files found in '{source_dir}'.")
        print("Place your documents there and re-run.")
        sys.exit(0)

    documents = []
    for file_path in files:
        try:
            if file_path.suffix == ".pdf":
                loader = PyPDFLoader(str(file_path))
            else:
                loader = TextLoader(str(file_path), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = file_path.name
            documents.extend(docs)
            print(f"  Loaded: {file_path.name} ({len(docs)} page(s))")
        except Exception as e:
            print(f"  Skipped: {file_path.name} — {e}")

    return documents


def ingest(source_dir: str = DOCUMENTS_DIR, upload: bool = False):
    start = time.time()

    print(f"\nRAG Knowledge Assistant — Ingestion")
    print(f"Source directory : {source_dir}")
    print(f"ChromaDB path    : {PERSIST_DIRECTORY}")
    print(f"Embedding model  : {EMBEDDING_MODEL}")
    print()

    # Step 1: Load
    print("Step 1/3 — Loading documents ...")
    documents = load_documents(source_dir)
    print(f"  Total documents loaded: {len(documents)}")

    # Step 2: Chunk
    print("\nStep 2/3 — Chunking ...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    avg_len = sum(len(c.page_content) for c in chunks) // max(len(chunks), 1)
    print(f"  Chunks created   : {len(chunks)}")
    print(f"  Avg chunk length : {avg_len} chars")

    # Step 3: Embed and index
    print("\nStep 3/3 — Embedding and indexing (this takes a minute) ...")
    embeddings  = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )

    elapsed = time.time() - start
    count   = vectorstore._collection.count()
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Total chunks indexed : {count}")
    print(f"Index saved to       : {PERSIST_DIRECTORY}/")

    # Optional GCS upload
    if upload:
        print("\nUploading index to GCS ...")
        from google.cloud import storage as gcs
        bucket_name = os.getenv("GCS_BUCKET", "rag-knowledge-assistant-index")
        try:
            client = gcs.Client()
            bucket = client.bucket(bucket_name)
            uploaded = 0
            for root, dirs, files in os.walk(PERSIST_DIRECTORY):
                for file in files:
                    local_file = os.path.join(root, file)
                    gcs_path   = "chroma_db/" + os.path.relpath(
                        local_file, PERSIST_DIRECTORY
                    ).replace("\\", "/")
                    bucket.blob(gcs_path).upload_from_filename(local_file)
                    uploaded += 1
            print(f"Uploaded {uploaded} files to gs://{bucket_name}/chroma_db/")
        except Exception as e:
            print(f"GCS upload failed: {e}")
            print("Index is saved locally. Run with GCP credentials to upload.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG knowledge base.")
    parser.add_argument(
        "--source", default=DOCUMENTS_DIR,
        help=f"Folder containing .txt and .pdf files (default: {DOCUMENTS_DIR})"
    )
    parser.add_argument(
        "--upload", action="store_true",
        help="Upload the resulting ChromaDB index to GCS after ingestion"
    )
    args = parser.parse_args()
    ingest(source_dir=args.source, upload=args.upload)
