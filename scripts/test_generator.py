import sys
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

# Load .env before importing to ensure model name is picked up
load_dotenv(project_root / ".env")

from rag_app.retrieval.retriever import RetrievalResult
from rag_app.generation.generator import generate_answer, get_gemini_model_name

def main():
    print(f"Configured Model: {get_gemini_model_name()}")
    
    dummy_retrieval: RetrievalResult = {
        "query": "What is photosynthesis?",
        "requested_k": 1,
        "retrieved_children": [],
        "corresponding_parents": [
            {
                "point_id": "p1",
                "content": "Photosynthesis is the process by which green plants make food.",
                "parent_id": "p1",
                "document_id": "doc1",
                "document_checksum": "hash1",
                "chunking_config_hash": "chash1",
                "parent_index": 0,
                "status": "ACTIVE",
                "metadata": {"source_file": "ch6 Life Processes.pdf"}
            }
        ],
        "retrieved_child_count": 1,
        "retrieved_parent_count": 1,
        "duration_seconds": 0.1
    }
    
    try:
        result = generate_answer(dummy_retrieval)
        print("Test Result: SUCCESS")
        print(f"Answer: {result['answer']}")
        print(f"Citations: {result['citations']}")
    except Exception as e:
        print(f"Test Result: FAILED_DUE_TO_CREDENTIALS - {e}")

if __name__ == "__main__":
    main()
