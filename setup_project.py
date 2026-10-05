import os

def create_structure():
    """
    Creates a production-grade directory structure for the NCERT Chatbot (Hybrid Architecture).
    """
    folders = [
        "configs",
        "data",
        "data/pdf",
        "data/pdf_archive",
        "data/faiss_index",
        "frontend",
        "frontend/components",
        "scripts",
        "tests",
        "tests/api",
        "tests/rag_app",
        "tests/cache",
        "tests/frontend",
        "api",
        "cache",
        "memory",
        "evaluation",
        "rag_app",
        "rag_app/ingestion",
        "rag_app/chunking",
        "rag_app/embeddings",
        "rag_app/storage",
        "rag_app/retrieval",
        "rag_app/generation",
        "rag_app/observability",
    ]

    files = [
        "configs/document_metadata.csv",
        "data/cache_seed.jsonl",
        "frontend/app.py",
        "frontend/requirements.txt",
        "scripts/ingest_pdfs.py",
        "scripts/seed_cache.py",
        "api/__init__.py",
        "api/routes.py",
        "api/schemas.py",
        "cache/__init__.py",
        "cache/smart_cache.py",
        "memory/__init__.py",
        "memory/session.py",
        "evaluation/__init__.py",
        "evaluation/ragas_eval.py",
        "rag_app/__init__.py",
        "rag_app/ingestion/__init__.py",
        "rag_app/ingestion/loader.py",
        "rag_app/chunking/__init__.py",
        "rag_app/chunking/chunker.py",
        "rag_app/embeddings/__init__.py",
        "rag_app/embeddings/embedder.py",
        "rag_app/storage/__init__.py",
        "rag_app/storage/sqlite_store.py",
        "rag_app/storage/faiss_store.py",
        "rag_app/retrieval/__init__.py",
        "rag_app/retrieval/retriever.py",
        "rag_app/generation/__init__.py",
        "rag_app/generation/generator.py",
        "rag_app/observability/__init__.py",
        "rag_app/observability/logger.py",
        "rag_app/observability/langsmith_tracer.py",
        "main.py",
        "requirements.txt",
        ".env.example",
        ".gitignore",
        "README.md"
    ]

    print("Initializing hybrid production folder structure...")

    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created directory: {folder}")

    for file in files:
        if not os.path.exists(file):
            with open(file, 'w', encoding='utf-8') as f:
                # Add basic content to some files
                if file.endswith("__init__.py"):
                    pass
                elif file == ".gitignore":
                    f.write("venv/\n__pycache__/\n.env\ndata/pdf/*\ndata/faiss_index/*\n.pytest_cache/\n")
                elif file == ".env.example":
                    f.write("GEMINI_API_KEY=your_key_here\n")
                elif file == "configs/document_metadata.csv":
                    f.write("filename,domain,document_type\n")
            print(f"Created file: {file}")
        else:
            print(f"File already exists (skipped): {file}")

    print("\nHybrid project structure initialized successfully.")

if __name__ == "__main__":
    create_structure()
