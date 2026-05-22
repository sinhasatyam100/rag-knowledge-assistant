# retriever.py
# Two-stage retrieval: fast bi-encoder → accurate cross-encoder reranker
# 
# Why two stages?
# Bi-encoder: embeds query and doc SEPARATELY, compares vectors
#   → Fast (one embedding per doc, pre-computed)
#   → Less accurate (misses nuanced relevance)
#
# Cross-encoder: reads query AND doc TOGETHER in one forward pass
#   → Slow (can't pre-compute, runs at query time)
#   → Much more accurate (sees full context of both)
#
# Solution: bi-encoder retrieves top-20 candidates cheaply,
# cross-encoder reranks them accurately, return top-k

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from sentence_transformers import CrossEncoder
from langchain_chroma import Chroma
from logs import logger
import time

# Load cross-encoder once at module level
# Same model every request — no reloading
# ms-marco-MiniLM-L-6-v2 is the standard choice:
# - Trained on MS MARCO passage ranking dataset
# - Small (22M params) → fast on CPU
# - High quality for passage reranking tasks
_cross_encoder = None

def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        logger.info("Loading cross-encoder model...")
        _cross_encoder = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        logger.info("Cross-encoder loaded")
    return _cross_encoder


def retrieve_and_rerank(
    query: str,
    vectorstore: Chroma,
    initial_k: int = 20,
    final_k: int = 4
) -> list:
    """
    Two-stage retrieval with reranking.
    
    Stage 1: Retrieve initial_k candidates using bi-encoder (ChromaDB)
    Stage 2: Rerank with cross-encoder, return top final_k
    
    Args:
        query: the user's question
        vectorstore: the ChromaDB instance from app.state
        initial_k: how many candidates to retrieve in stage 1
        final_k: how many to return after reranking
    
    Returns:
        List of top final_k documents, reranked by relevance
    """
    start = time.time()

    # Stage 1 — fast bi-encoder retrieval
    # Retrieve more than we need so reranker has candidates to work with
    # Rule of thumb: initial_k = 4-5x final_k
    candidates = vectorstore.similarity_search(query, k=initial_k)

    if not candidates:
        return []

    if len(candidates) <= final_k:
        # Not enough candidates to bother reranking
        logger.info(f"Only {len(candidates)} candidates, skipping rerank")
        return candidates

    stage1_time = (time.time() - start) * 1000
    logger.info(f"Stage 1: Retrieved {len(candidates)} candidates in {stage1_time:.0f}ms")

    # Stage 2 — cross-encoder reranking
    # CrossEncoder takes (query, passage) pairs and scores them 0-1
    # Higher score = more relevant to the query
    reranker = get_cross_encoder()

    # Build (query, document_text) pairs for the cross-encoder
    pairs = [(query, doc.page_content) for doc in candidates]

    # Scores: array of floats, one per pair
    # Cross-encoder reads BOTH query and doc together — much more accurate
    scores = reranker.predict(pairs)

    # Zip scores with documents, sort by score descending
    scored_docs = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True
    )

    # Take top final_k
    top_docs = [doc for _, doc in scored_docs[:final_k]]
    top_scores = [round(float(score), 4) for score, _ in scored_docs[:final_k]]

    stage2_time = (time.time() - start) * 1000
    logger.info(
        f"Stage 2: Reranked to top {final_k} in {stage2_time:.0f}ms total. "
        f"Top scores: {top_scores}"
    )

    return top_docs