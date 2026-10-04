import os

def create_structure():
    """
    Creates a production-grade directory structure for the NCERT Chatbot.
    """
    folders = [
        "backend",
        "backend/api",
        "backend/core",
        "backend/rag",
        "backend/cache",
        "backend/observability",
        "backend/evaluation",
        "backend/memory",
        "frontend",
        "frontend/components",
        "data",
        "data/raw",
        "data/faiss_index",
        "scripts",
        "tests",
        "tests/backend",
        "tests/frontend"
    ]

    files = [
        "backend/__init__.py",
        "backend/api/__init__.py",
        "backend/api/routes.py",
        "backend/api/schemas.py",
        "backend/core/__init__.py",
        "backend/core/config.py",
        "backend/core/logger.py",
        "backend/rag/__init__.py",
        "backend/rag/loader.py",
        "backend/rag/chunker.py",
        "backend/rag/embedder.py",
        "backend/rag/vector_store.py",
        "backend/rag/generator.py",
        "backend/cache/__init__.py",
        "backend/cache/smart_cache.py",
        "backend/observability/__init__.py",
        "backend/observability/langsmith_tracer.py",
        "backend/evaluation/__init__.py",
        "backend/evaluation/ragas_eval.py",
        "backend/memory/__init__.py",
        "backend/memory/session.py",
        "scripts/ingest_pdfs.py",
        "scripts/seed_cache.py",
        "data/cache_seed.jsonl",
        "backend/main.py",
        "backend/requirements.txt",
        "frontend/app.py",
        "frontend/requirements.txt",
        ".env.example",
        ".gitignore",
        "README.md"
    ]

    print("Initializing production folder structure...")

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
                    f.write("venv/\n__pycache__/\n.env\ndata/raw/*\ndata/faiss_index/*\n.pytest_cache/\n")
                elif file == ".env.example":
                    f.write("GEMINI_API_KEY=your_key_here\n")
            print(f"Created file: {file}")
        else:
            print(f"File already exists (skipped): {file}")

    print("\nProject structure initialized successfully.")

if __name__ == "__main__":
    create_structure()
