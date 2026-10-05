import time
import uuid
from typing import Any
from rag_app.retrieval.retriever import retrieve_context
from rag_app.generation.generator import generate_answer
from rag_app.observability.logger import logger
from rag_app.storage.app_store import (
    init_app_tables, get_session_history, update_session_history,
    get_cached_response, set_cached_response
)

# Ensure tables exist
init_app_tables()

def generate_session_id() -> str:
    return str(uuid.uuid4())

def rewrite_query_with_history(query: str, history: list[dict]) -> str:
    if not history:
        return query
    
    context = []
    # Take last 2 turns to prevent massive prompt expansion
    for msg in history[-2:]:
        context.append(f"{msg['role'].capitalize()}: {msg['content']}")
    
    context_str = "\n".join(context)
    return f"Context of conversation:\n{context_str}\n\nCurrent Question: {query}"

def chat(session_id: str, message: str) -> dict[str, Any]:
    start_time = time.perf_counter()
    
    if not message or not message.strip():
        raise ValueError("Message cannot be empty.")
        
    logger.info(f"Orchestrator received message for session {session_id}")
    
    # 1. Smart Cache Lookup
    cached = get_cached_response(message.strip().lower())
    if cached:
        logger.info("Cache HIT.")
        cached["cache_hit"] = True
        cached["latency_ms"] = int((time.perf_counter() - start_time) * 1000)
        
        # Update session history even on cache hit
        history = get_session_history(session_id)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": cached["reply"]})
        update_session_history(session_id, history)
        
        return cached

    logger.info("Cache MISS.")
    
    # 2. Session Context Injection
    history = get_session_history(session_id)
    search_query = rewrite_query_with_history(message, history)
    
    # 3. FAISS + SQLite + CrossEncoder Retrieval
    try:
        retrieval_result = retrieve_context(query=search_query, k=6, candidate_k=15)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise RuntimeError("Failed to retrieve context.")

    # 4. Grounded Gemini Generation
    try:
        gen_result = generate_answer(retrieval_result)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise RuntimeError("Failed to generate answer.")
    
    reply = gen_result.get("answer", "I could not generate an answer.")
    citations = gen_result.get("citations", [])
    
    formatted_citations = []
    if citations:
        for c in citations:
            source = c.get("source_file") or "Unknown Document"
            formatted_citations.append({"label": c["label"], "source": source})

    response_data = {
        "reply": reply,
        "citations": formatted_citations,
        "cache_hit": False
    }
    
    # 5. Cache Storage (Only if answer was actually grounded in context)
    if not gen_result.get("no_context", True):
        set_cached_response(message.strip().lower(), response_data)
        
    # 6. Update SQLite Session History
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    update_session_history(session_id, history)
    
    response_data["latency_ms"] = int((time.perf_counter() - start_time) * 1000)
    return response_data
