"""
retriever.py — Two-stage retrieval pipeline.

Stage 1 (bi-encoder): ChromaDB similarity search retrieves initial_k candidates.
    Fast — embeddings are pre-computed at ingestion time.
    Less accurate — encodes query and document independently.

Stage 2 (cross-encoder): Reranks candidates by relevance, returns top final_k.
    Slower — reads query and document together in one forward pass.
    Much more accurate — sees the full interaction between query and document.

Why two stages?
    Running the cross-encoder on every document in the corpus is too slow.
    Running only the bi-encoder misses nuanced relevance.
    Combining both gives speed and accuracy: fetch 20 cheaply, rerank accurately.
"""

import time
from sentence_transformers import CrossEncoder
from langchain_chroma import Chroma
from logs import logger


# ── Cross-encoder singleton ───────────────────────────────────────────
# Loaded once at startup (called from lifespan and worker init).
# ms-marco-MiniLM-L-6-v2: trained on MS MARCO passage ranking.
# 22M parameters — fast on CPU. High quality for reranking.

_cross_encoder = None


def get_cross_encoder() -> CrossEncoder:
    """Return the cross-encoder model, loading it on first call."""
    global _cross_encoder
    if _cross_encoder is None:
        logger.info("Loading cross-encoder model ...")
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("Cross-encoder loaded.")
    return _cross_encoder


# ── Two-stage retrieval ───────────────────────────────────────────────

def retrieve_and_rerank(
    query:      str,
    vectorstore: Chroma,
    initial_k:  int = 20,
    final_k:    int = 4,
) -> list:
    """
    Two-stage retrieval with cross-encoder reranking.

    Args:
        query:       The user's question.
        vectorstore: ChromaDB instance from app.state.
        initial_k:   Candidates to fetch in stage 1 (bi-encoder).
                     Rule of thumb: 4-5x final_k.
        final_k:     Documents to return after reranking.

    Returns:
        List of up to final_k Documents, ordered by relevance.
    """
    start = time.time()

    # Stage 1 — bi-encoder retrieval
    candidates = vectorstore.similarity_search(query, k=initial_k)

    if not candidates:
        logger.info("Retrieval returned 0 candidates.")
        return []

    if len(candidates) <= final_k:
        logger.info(f"Only {len(candidates)} candidates — skipping rerank.")
        return candidates

    stage1_ms = (time.time() - start) * 1000
    logger.info(f"Stage 1: {len(candidates)} candidates in {stage1_ms:.0f}ms")

    # Stage 2 — cross-encoder reranking
    reranker = get_cross_encoder()
    pairs    = [(query, doc.page_content) for doc in candidates]
    scores   = reranker.predict(pairs)

    scored_docs = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    top_docs    = [doc for _, doc in scored_docs[:final_k]]
    top_scores  = [round(float(s), 4) for s, _ in scored_docs[:final_k]]

    total_ms = (time.time() - start) * 1000
    logger.info(
        f"Stage 2: top {final_k} in {total_ms:.0f}ms total. Scores: {top_scores}"
    )

    return top_docs
