import sqlite3
import time
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from rag_app.embeddings.embedder import ChunkEmbedder
from rag_app.storage.faiss_store import FAISSIndex
from rag_app.observability.logger import logger

DB_PATH = project_root / "data" / "rag.db"
FAISS_INDEX_PATH = project_root / "data" / "faiss.index"
FAISS_MAPPING_PATH = project_root / "data" / "faiss_mapping.pkl"

def get_unembedded_children(conn):
    cur = conn.cursor()
    # ONLY fetch children. Never fetch parents.
    cur.execute("SELECT child_id, text FROM children")
    return cur.fetchall()

def validate_test_query(embedder):
    query = "What is a chemical reaction?"
    vec = embedder.embed_texts([query])
    assert len(vec) == 1
    assert len(vec[0]) == embedder.get_dimension()
    logger.info("Test query embedded successfully with the same model.")

def main():
    logger.info("Starting embedding phase...")
    start_time = time.perf_counter()
    
    if not DB_PATH.exists():
        logger.error("SQLite database not found. Run ingest_pdfs.py first.")
        return
        
    conn = sqlite3.connect(DB_PATH)
    children = get_unembedded_children(conn)
    conn.close()
    
    if not children:
        logger.warning("No children found to embed.")
        return
        
    logger.info(f"Loaded {len(children)} child chunks from SQLite.")
    
    embedder = ChunkEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    dimension = embedder.get_dimension()
    logger.info(f"Loaded embedding model with dimension: {dimension}")
    
    validate_test_query(embedder)
    
    faiss_store = FAISSIndex(dimension, FAISS_INDEX_PATH, FAISS_MAPPING_PATH)
    
    existing_ids = set(faiss_store.index_to_child_id.values())
    to_embed = [(cid, text) for cid, text in children if cid not in existing_ids]
    
    if not to_embed:
        logger.info("All child chunks are already embedded.")
        return
        
    logger.info(f"Embedding {len(to_embed)} new children (ignoring {len(existing_ids)} existing)...")
    
    batch_size = 128
    for i in range(0, len(to_embed), batch_size):
        batch = to_embed[i:i+batch_size]
        child_ids = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        
        vectors = embedder.embed_texts(texts)
        faiss_store.add_vectors(vectors, child_ids)
        
        if (i // batch_size + 1) % 5 == 0:
            logger.info(f"Embedded batch {i//batch_size + 1}, total vectors so far: {faiss_store.index.ntotal}")
            
    # Final log if didn't hit modulo
    if (len(to_embed) // batch_size + 1) % 5 != 0:
         logger.info(f"Finished embedding all batches. Total vectors: {faiss_store.index.ntotal}")
        
    faiss_store.save()
    
    duration = time.perf_counter() - start_time
    logger.info(f"Embedding completed. Total vectors in FAISS: {faiss_store.index.ntotal}. Duration: {duration:.3f}s")
    
    # Validation
    assert faiss_store.index.ntotal == len(faiss_store.index_to_child_id), "Vector count does not match mapping count!"
    assert faiss_store.index.d == dimension, "Vector dimension mismatch!"
    
    logger.info("Validation passed: Vector count exactly matches child_id mapping count. Dimension matches. Only children embedded.")

if __name__ == "__main__":
    main()
