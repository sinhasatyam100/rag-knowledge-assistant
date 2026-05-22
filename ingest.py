# ingest.py

import os
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import bs4
import time

# ── Configuration ────────────────────────────────────────────────────
# Change these to change what gets indexed.

URLS = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Natural_language_processing",
]

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def ingest():
    start = time.time()
    
    # ── Step 1: Load ─────────────────────────────────────────────────
    print(f"Loading {len(URLS)} URLs...")
    
    loader = WebBaseLoader(
        web_paths=URLS,
        bs_kwargs=dict(parse_only=bs4.SoupStrainer("p"))
    )
    
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    
    # ── Step 2: Chunk ─────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    print(f"Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")
    
    # ── Step 3: Embed + Index ─────────────────────────────────────────
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    print("Embedding and indexing chunks — this takes a minute...")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    
    elapsed = time.time() - start
    total_indexed = vectorstore._collection.count()
        
    try:
        total_indexed = vectorstore._collection.count()
        print(f"\nDone in {elapsed:.1f}s")
        print(f"Total chunks indexed: {total_indexed}")
        print(f"Index saved to: {PERSIST_DIRECTORY}/")
    except Exception as e:
        print(f"\nDone in {elapsed:.1f}s")
        print(f"WARNING: Could not verify count: {e}")
        print(f"Check {PERSIST_DIRECTORY}/ folder to confirm data was saved")


if __name__ == "__main__":
    ingest()