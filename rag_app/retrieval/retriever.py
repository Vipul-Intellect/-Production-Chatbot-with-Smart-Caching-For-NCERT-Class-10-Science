from time import perf_counter
from typing import Any, TypedDict
from pathlib import Path
import pickle

import faiss
import numpy as np
from sentence_transformers import CrossEncoder

from rag_app.embeddings.embedder import ChunkEmbedder
from rag_app.observability.logger import logger
from rag_app.storage.sqlite_store import fetch_children_by_ids, fetch_parents_by_ids

class RetrievedChild(TypedDict):
    point_id: str
    score: float
    rerank_score: float
    content: str | None
    child_id: Any
    parent_id: Any
    document_id: Any
    document_checksum: Any
    chunking_config_hash: Any
    parent_index: Any
    child_index: Any
    status: Any
    metadata: dict[str, Any]

class RetrievedParent(TypedDict):
    point_id: str
    content: str | None
    parent_id: Any
    document_id: Any
    document_checksum: Any
    chunking_config_hash: Any
    parent_index: Any
    status: Any
    metadata: dict[str, Any]

class RetrievalResult(TypedDict):
    query: str
    requested_k: int
    retrieved_children: list[RetrievedChild]
    corresponding_parents: list[RetrievedParent]
    retrieved_child_count: int
    retrieved_parent_count: int
    duration_seconds: float


# Singleton instances to avoid reloading expensive models/indexes per request
_EMBEDDER = None
_RERANKER = None
_FAISS_INDEX = None
_FAISS_MAPPING = None

PROJECT_ROOT = Path(__file__).parent.parent.parent
FAISS_INDEX_PATH = PROJECT_ROOT / "data" / "faiss.index"
FAISS_MAPPING_PATH = PROJECT_ROOT / "data" / "faiss_mapping.pkl"

def _get_embedder() -> ChunkEmbedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = ChunkEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    return _EMBEDDER

def _get_reranker() -> CrossEncoder:
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _RERANKER

def _get_faiss_index() -> faiss.Index:
    global _FAISS_INDEX
    if _FAISS_INDEX is None:
        if not FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}. Run embed_chunks.py first.")
        _FAISS_INDEX = faiss.read_index(str(FAISS_INDEX_PATH))
    return _FAISS_INDEX

def _get_faiss_mapping() -> dict[int, str]:
    global _FAISS_MAPPING
    if _FAISS_MAPPING is None:
        if not FAISS_MAPPING_PATH.exists():
            raise FileNotFoundError(f"FAISS mapping not found at {FAISS_MAPPING_PATH}. Run embed_chunks.py first.")
        with open(FAISS_MAPPING_PATH, 'rb') as f:
            _FAISS_MAPPING = pickle.load(f)
    return _FAISS_MAPPING


def _row_to_metadata(row: dict[str, Any]) -> dict[str, Any]:
    # Everything except the raw text
    return {k: v for k, v in row.items() if k != "text"}

def retrieve_context(
    query: str,
    k: int,
    candidate_k: int = 20,
) -> RetrievalResult:
    """Retrieve ACTIVE child chunks via FAISS -> SQLite and rerank with cross-encoder."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Retrieval query must be a non-empty string.")
    if k <= 0:
        raise ValueError("Retrieval k must be greater than 0.")
    if candidate_k < k:
        candidate_k = k

    start_time = perf_counter()
    logger.info(f"Retrieval started. query={query!r} candidate_k={candidate_k} final_k={k}")

    # 1. Embed query
    embedder = _get_embedder()
    query_vector = embedder.embed_texts([query])[0]
    
    faiss_index = _get_faiss_index()
    if len(query_vector) != faiss_index.d:
        raise ValueError(f"Query embedding dimension mismatch: expected {faiss_index.d}, got {len(query_vector)}")

    # 2. FAISS Candidate Search
    query_np = np.array([query_vector], dtype=np.float32)
    # L2 distance: lower distance = higher similarity
    distances, indices = faiss_index.search(query_np, candidate_k)
    
    mapping = _get_faiss_mapping()
    candidate_child_ids = []
    child_id_to_score = {}
    
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        child_id = mapping.get(idx)
        if child_id:
            candidate_child_ids.append(child_id)
            child_id_to_score[child_id] = float(dist)

    # 3. SQLite Lookup (Ensures only ACTIVE chunks are loaded and returns full text/metadata)
    children_rows = fetch_children_by_ids(candidate_child_ids)
    
    # 4. Cross-Encoder Reranking
    retrieved_children = []
    if children_rows:
        reranker = _get_reranker()
        cross_input = [(query, row["text"]) for row in children_rows]
        rerank_scores = reranker.predict(cross_input)
        
        for row, r_score in zip(children_rows, rerank_scores):
            child_id = row["child_id"]
            metadata = _row_to_metadata(row)
            retrieved_children.append({
                "point_id": child_id, # Alias for orchestrator compatibility
                "score": child_id_to_score.get(child_id, 0.0), # Raw FAISS distance
                "rerank_score": float(r_score),
                "content": row.get("text"),
                "child_id": child_id,
                "parent_id": row.get("parent_id"),
                "document_id": row.get("document_id"),
                "document_checksum": row.get("document_checksum"),
                "chunking_config_hash": row.get("chunking_config_hash"),
                "parent_index": row.get("parent_index"),
                "child_index": row.get("child_index"),
                "status": row.get("status"),
                "metadata": metadata
            })
            
        # Sort by rerank score descending (higher is better for CrossEncoder)
        retrieved_children.sort(key=lambda x: x["rerank_score"], reverse=True)
        retrieved_children = retrieved_children[:k]
    
    # 5. Parent Context Retrieval
    parent_ids = list(dict.fromkeys(child["parent_id"] for child in retrieved_children if child.get("parent_id")))
    parent_rows = fetch_parents_by_ids(parent_ids)
    
    parent_row_map = {row["parent_id"]: row for row in parent_rows}
    
    corresponding_parents = []
    for pid in parent_ids:
        if pid in parent_row_map:
            row = parent_row_map[pid]
            metadata = _row_to_metadata(row)
            corresponding_parents.append({
                "point_id": pid,
                "content": row.get("text"),
                "parent_id": pid,
                "document_id": row.get("document_id"),
                "document_checksum": row.get("document_checksum"),
                "chunking_config_hash": row.get("chunking_config_hash"),
                "parent_index": row.get("parent_index"),
                "status": row.get("status"),
                "metadata": metadata
            })
        else:
            logger.warning(f"Retrieved child referenced missing or INACTIVE parent_id={pid}")

    duration_seconds = perf_counter() - start_time
    if not retrieved_children:
        logger.info("Retrieval completed with no ACTIVE child results.")
    else:
        logger.info(
            "Retrieval completed. child_results=%s parent_results=%s duration=%.3fs",
            len(retrieved_children),
            len(corresponding_parents),
            duration_seconds,
        )

    return {
        "query": query,
        "requested_k": k,
        "retrieved_children": retrieved_children,
        "corresponding_parents": corresponding_parents,
        "retrieved_child_count": len(retrieved_children),
        "retrieved_parent_count": len(corresponding_parents),
        "duration_seconds": duration_seconds,
    }
