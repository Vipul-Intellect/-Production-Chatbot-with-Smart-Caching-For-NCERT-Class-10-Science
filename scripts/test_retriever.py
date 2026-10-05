import sqlite3
import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from rag_app.retrieval.retriever import retrieve_context

DB_PATH = project_root / "data" / "rag.db"

def main():
    # 1. Activate all chunks for testing
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE children SET status = 'ACTIVE'")
    conn.execute("UPDATE parents SET status = 'ACTIVE'")
    conn.commit()
    conn.close()
    
    print("Running retrieval...")
    result = retrieve_context("What is a chemical reaction?", k=3, candidate_k=15)
    
    # Validation checks
    assert result['retrieved_child_count'] == 3, f"Expected 3 children, got {result['retrieved_child_count']}"
    assert result['retrieved_parent_count'] <= 3, "Too many parents retrieved"
    
    first = result['retrieved_children'][0]
    print(f"Top child rerank_score: {first['rerank_score']}")
    print(f"Top child FAISS L2 score (lower is better): {first['score']}")
    
    assert "rerank_score" in first
    assert "score" in first
    assert first["content"] is not None
    assert first["parent_id"] is not None
    
    parents = result['corresponding_parents']
    assert len(parents) > 0, "No parents returned"
    
    parent_ids_from_children = set(c["parent_id"] for c in result["retrieved_children"])
    parent_ids_from_parents = set(p["parent_id"] for p in parents)
    assert parent_ids_from_children.issubset(parent_ids_from_parents) or parent_ids_from_children == parent_ids_from_parents, "Parent mapping mismatch"
    
    # 13. Empty query test
    try:
        retrieve_context("", 3)
        assert False, "Empty query did not raise exception"
    except ValueError:
        print("Empty query handled correctly.")
        
    print("All validations passed.")
    
if __name__ == "__main__":
    main()
