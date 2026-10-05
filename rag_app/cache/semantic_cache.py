import os
import re
import uuid
import numpy as np
from time import perf_counter
from pathlib import Path

from rag_app.observability.logger import logger
from rag_app.storage.faiss_store import FAISSIndex
from rag_app.storage.app_store import get_cached_response, set_cached_response, get_redis_client
from rag_app.retrieval.retriever import _get_embedder, _get_reranker

PROJECT_ROOT = Path(__file__).parent.parent.parent
CACHE_FAISS_INDEX_PATH = PROJECT_ROOT / "data" / "cache_faiss.index"
CACHE_FAISS_MAPPING_PATH = PROJECT_ROOT / "data" / "cache_faiss_mapping.pkl"

_CACHE_FAISS_INDEX = None

def _get_cache_faiss() -> FAISSIndex:
    global _CACHE_FAISS_INDEX
    if _CACHE_FAISS_INDEX is None:
        embedder = _get_embedder()
        _CACHE_FAISS_INDEX = FAISSIndex(embedder.get_dimension(), CACHE_FAISS_INDEX_PATH, CACHE_FAISS_MAPPING_PATH)
    return _CACHE_FAISS_INDEX

def extract_numbers(text: str) -> set[float]:
    matches = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    return set(float(m) for m in matches)

def is_context_dependent(query: str) -> bool:
    lower_q = query.lower()
    # Pronouns indicating follow-up that depend on conversation
    context_words = {"it", "this", "that", "these", "those", "he", "she", "they", "about", "more", "simply", "previous"}
    words = set(re.findall(r'\b\w+\b', lower_q))
    if words.intersection(context_words) or len(words) < 3:
        return True
    return False

def has_contradictory_terms(q1: str, q2: str) -> bool:
    pairs = [
        ({"concave"}, {"convex"}),
        ({"myopia", "nearsightedness"}, {"hypermetropia", "farsightedness"}),
        ({"real"}, {"virtual"}),
        ({"converging"}, {"diverging"}),
        ({"series"}, {"parallel"}),
        ({"oxidation"}, {"reduction"}),
        ({"exothermic"}, {"endothermic"})
    ]
    w1 = set(re.findall(r'\b[a-z]+\b', q1.lower()))
    w2 = set(re.findall(r'\b[a-z]+\b', q2.lower()))
    
    for a, b in pairs:
        if (w1.intersection(a) and w2.intersection(b)) or (w1.intersection(b) and w2.intersection(a)):
            return True
    return False

def check_semantic_cache(query: str) -> dict | None:
    start_time = perf_counter()
    if is_context_dependent(query):
        logger.info("Cache bypassed: Context-dependent query detected.")
        return None

    query_numbers = extract_numbers(query)
    embedder = _get_embedder()
    faiss_store = _get_cache_faiss()
    
    if faiss_store.index.ntotal == 0:
        return None
        
    query_vector = embedder.embed_texts([query])[0]
    query_np = np.array([query_vector], dtype=np.float32)
    
    # Search Top K=5 to check the best candidate
    distances, indices = faiss_store.index.search(query_np, min(5, faiss_store.index.ntotal))
    
    best_cache_id = None
    
    mapping = faiss_store.index_to_child_id
    
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
            
        cache_id = mapping.get(idx)
        if not cache_id:
            continue
            
        from rag_app.storage.app_store import get_cached_question
        cached_q = get_cached_question(cache_id)
        if not cached_q:
            continue
            
        cached_numbers = extract_numbers(cached_q)
        if query_numbers != cached_numbers:
            logger.info(f"Cache miss candidate due to numerical parameter mismatch: {query_numbers} vs {cached_numbers}")
            continue
            
        if has_contradictory_terms(query, cached_q):
            logger.info(f"Cache miss candidate due to contradictory terms in '{query}' vs '{cached_q}'")
            continue
            
        # CrossEncoder safety check for semantic correctness
        reranker = _get_reranker()
        cross_score = float(reranker.predict([(query, cached_q)])[0])
        
        if cross_score > 2.0:
            best_cache_id = cache_id
            break
        else:
            logger.info(f"Cache candidate rejected by reranker score ({cross_score:.2f}) for '{cached_q}'")

    if best_cache_id:
        response = get_cached_response(best_cache_id)
        if response:
            latency = (perf_counter() - start_time) * 1000
            response["latency_ms"] = int(latency)
            response["cache_hit"] = True
            logger.info(f"Semantic Cache HIT. Latency: {latency:.1f}ms")
            return response
            
    return None

def write_to_semantic_cache(query: str, response: dict):
    if is_context_dependent(query):
        return
        
    cache_id = str(uuid.uuid4())
    
    from rag_app.storage.app_store import set_cached_question
    set_cached_question(cache_id, query)
    
    # Save the response dict
    set_cached_response(cache_id, response)
    
    # Embed and add to FAISS
    embedder = _get_embedder()
    faiss_store = _get_cache_faiss()
    vector = embedder.embed_texts([query])[0]
    faiss_store.add_vectors([vector], [cache_id])
    faiss_store.save()
    logger.info(f"Written new question to Semantic Cache: {query}")
