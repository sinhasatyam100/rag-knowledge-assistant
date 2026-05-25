"""
cache.py — Semantic cache backed by Redis.

Standard caches match exact query strings.
Semantic cache matches meaning — "What is ML?" and "Explain machine learning"
are different strings but semantically identical questions, so they share
a cached answer.

How it works:
1. Embed the incoming query with the same MiniLM model used for retrieval.
2. Scan cached query embeddings in Redis.
3. If any cached query has cosine similarity >= SIMILARITY_THRESHOLD, return
   its answer immediately — skipping retrieval and LLM entirely.
4. On a cache miss, run the full pipeline and store the result with a TTL.

Redis is already in the stack (Celery broker). In production it runs as
Cloud Memorystore — same Redis, managed by GCP.
"""

import os
import json
import time
import hashlib

import numpy as np
import redis

from logs import logger

# ── Configuration ─────────────────────────────────────────────────────

CACHE_TTL_SECONDS    = int(os.getenv("CACHE_TTL_SECONDS",         "3600"))
SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.95"))

# TTL guide:
#   Static knowledge bases (policies, runbooks): 86400 (24 hours)
#   Regularly updated docs (tickets, wikis):     3600  (1 hour)   <- default
#   Live or frequently changing data:            300   (5 minutes)

# Threshold guide:
#   0.95 = very strict — near-identical questions share cache (few false positives)
#   0.90 = more aggressive — more hits, small risk of a wrong answer
#   Start at 0.95 and tune down if hit rate is too low.


# ── Redis client ──────────────────────────────────────────────────────

_redis_client = None


def get_redis_client() -> redis.Redis:
    """Return a Redis client, creating it once and reusing."""
    global _redis_client
    if _redis_client is None:
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        _redis_client = redis.from_url(url, decode_responses=True)
        logger.info(f"Redis client connected: {url[:40]}...")
    return _redis_client


# ── Math ──────────────────────────────────────────────────────────────

def cosine_similarity(a: list, b: list) -> float:
    """Cosine similarity between two embedding vectors."""
    a, b     = np.array(a), np.array(b)
    norm_a   = np.linalg.norm(a)
    norm_b   = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ── Cache operations ──────────────────────────────────────────────────

def cache_lookup(query_embedding: list) -> dict | None:
    """
    Find a cached answer whose query is semantically similar to query_embedding.

    Returns the cached result dict if similarity >= SIMILARITY_THRESHOLD, else None.

    Complexity: O(n) linear scan over cached entries.
    Sufficient for hundreds of cached queries.
    For millions of queries, switch to Redis Stack with HNSW vector indexing.
    """
    r = get_redis_client()
    try:
        keys = list(r.scan_iter("semcache:*"))
        if not keys:
            return None

        best_sim    = 0.0
        best_result = None

        for key in keys:
            raw = r.get(key)
            if not raw:
                continue
            try:
                cached = json.loads(raw)
                sim    = cosine_similarity(query_embedding, cached["embedding"])
                if sim > best_sim:
                    best_sim    = sim
                    best_result = cached
            except (json.JSONDecodeError, KeyError):
                continue

        if best_sim >= SIMILARITY_THRESHOLD:
            logger.info(f"Cache HIT  — similarity: {best_sim:.4f}")
            return best_result

        if best_sim > 0:
            logger.info(f"Cache MISS — best similarity: {best_sim:.4f}")
        return None

    except redis.RedisError as e:
        logger.warning(f"Cache lookup failed (Redis error): {e}. Continuing without cache.")
        return None


def cache_store(query: str, query_embedding: list, result: dict) -> bool:
    """
    Store a query result in the cache with automatic TTL expiry.

    Key: semcache:{md5(query)}
    Value: JSON with embedding, answer, sources, metadata
    TTL: CACHE_TTL_SECONDS (Redis auto-deletes after expiry)
    """
    r = get_redis_client()
    try:
        key   = f"semcache:{hashlib.md5(query.encode()).hexdigest()}"
        entry = {
            "query":           query,
            "embedding":       query_embedding,
            "answer":          result.get("answer", ""),
            "sources":         result.get("sources", []),
            "retrieval_count": result.get("retrieval_count", 0),
            "cached_at":       time.time(),
        }
        r.setex(key, CACHE_TTL_SECONDS, json.dumps(entry))
        logger.info(f"Cache STORED — TTL: {CACHE_TTL_SECONDS}s")
        return True

    except redis.RedisError as e:
        logger.warning(f"Cache store failed (Redis error): {e}. Result not cached.")
        return False


def cache_stats() -> dict:
    """
    Return cache statistics for the /health endpoint.
    Keys use 'entries' and 'hit_rate_pct' to match what the Streamlit UI reads.
    """
    r = get_redis_client()
    try:
        keys     = list(r.scan_iter("semcache:*"))
        entries  = len(keys)
        return {
            "entries":             entries,
            "hit_rate_pct":        0,       # Live hit rate would need a counter; 0 is safe default
            "ttl_seconds":         CACHE_TTL_SECONDS,
            "similarity_threshold": SIMILARITY_THRESHOLD,
        }
    except redis.RedisError:
        return {"entries": 0, "error": "Redis unavailable"}
