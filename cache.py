# cache.py
# Semantic cache using Redis.
#
# Why semantic instead of exact-match?
# "What is machine learning?" and "explain ML to me" are different strings
# but semantically identical. Exact-match cache misses this.
# Semantic cache embeds both queries, finds they're close (cosine > 0.95),
# and returns the cached answer — saving a full RAG pipeline execution.
#
# Why Redis?
# Already in our stack. Fast in-memory storage. Built-in TTL (expiry).
# In production: Cloud Memorystore — same Redis, managed by GCP.

import os
import json
import time
import hashlib
import numpy as np
import redis
from logs import logger

# TTL = how long cached answers live before expiring
# 3600 = 1 hour. Tune based on how often your knowledge base changes.
# For static docs: 86400 (24 hours). For news/live data: 300 (5 minutes).
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))

# Similarity threshold — how similar two queries must be to use the cache
# 0.95 = very similar, few false positives
# 0.90 = more aggressive caching, occasional wrong answers
# Start at 0.95 and tune down if cache hit rate is too low
SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.95"))

_redis_client = None

def get_redis_client() -> redis.Redis:
    """Get Redis client, connecting once and reusing."""
    global _redis_client
    if _redis_client is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        _redis_client = redis.from_url(redis_url, decode_responses=True)
        logger.info(f"Connected to Redis for semantic cache: {redis_url[:30]}...")
    return _redis_client


def cosine_similarity(a: list, b: list) -> float:
    """Compute cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cache_lookup(query_embedding: list) -> dict | None:
    """
    Look for a semantically similar cached query.
    
    Returns the cached result dict if similarity > threshold, else None.
    
    How it works:
    1. Scan all cached query embeddings in Redis
    2. Compute cosine similarity with the new query embedding
    3. If any cached query is similar enough, return its cached answer
    
    Time complexity: O(n) where n = number of cached queries
    For production with millions of queries, use Redis with vector 
    search (Redis Stack) instead of scanning.
    For our scale (hundreds of queries), linear scan is fine.
    """
    r = get_redis_client()
    
    try:
        # All our cache keys start with "semcache:"
        keys = list(r.scan_iter("semcache:*"))
        
        if not keys:
            return None
        
        best_similarity = 0.0
        best_result = None
        
        for key in keys:
            cached_data = r.get(key)
            if not cached_data:
                continue
            
            try:
                cached = json.loads(cached_data)
                similarity = cosine_similarity(
                    query_embedding,
                    cached["embedding"]
                )
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_result = cached
                    
            except (json.JSONDecodeError, KeyError):
                # Corrupted cache entry — skip it
                continue
        
        if best_similarity >= SIMILARITY_THRESHOLD:
            logger.info(
                f"Cache HIT — similarity: {best_similarity:.4f} "
                f"(threshold: {SIMILARITY_THRESHOLD})"
            )
            return best_result
        
        if best_similarity > 0:
            logger.info(
                f"Cache MISS — best similarity: {best_similarity:.4f} "
                f"(below threshold: {SIMILARITY_THRESHOLD})"
            )
        return None
        
    except redis.RedisError as e:
        # Cache errors should NEVER break the main application
        # Log and return None — app continues without cache
        logger.warning(f"Redis cache lookup failed: {e}. Proceeding without cache.")
        return None


def cache_store(query: str, query_embedding: list, result: dict) -> bool:
    """
    Store a query result in the semantic cache.
    
    Key: semcache:{hash of query}
    Value: JSON with embedding + result + metadata
    TTL: CACHE_TTL_SECONDS
    """
    r = get_redis_client()
    
    try:
        # Deterministic key from query text
        # Same query always maps to same key — enables exact-match overwrite
        key = f"semcache:{hashlib.md5(query.encode()).hexdigest()}"
        
        cache_entry = {
            "query": query,
            "embedding": query_embedding,
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "retrieval_count": result.get("retrieval_count", 0),
            "cached_at": time.time()
        }
        
        r.setex(
            key,
            CACHE_TTL_SECONDS,
            json.dumps(cache_entry)
        )
        
        logger.info(f"Cache STORED — key: {key[:30]}... TTL: {CACHE_TTL_SECONDS}s")
        return True
        
    except redis.RedisError as e:
        logger.warning(f"Redis cache store failed: {e}. Result not cached.")
        return False


def cache_stats() -> dict:
    """Return cache statistics — useful for /health endpoint."""
    r = get_redis_client()
    try:
        keys = list(r.scan_iter("semcache:*"))
        return {
            "cached_queries": len(keys),
            "ttl_seconds": CACHE_TTL_SECONDS,
            "similarity_threshold": SIMILARITY_THRESHOLD
        }
    except redis.RedisError:
        return {"cached_queries": 0, "error": "Redis unavailable"}