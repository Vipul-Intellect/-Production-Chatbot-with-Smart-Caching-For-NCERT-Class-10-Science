import sqlite3
import time
from pathlib import Path
import sys
import argparse

project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from rag_app.ingestion.loader import load_documents
from rag_app.chunking.chunker import create_parent_child_chunks
from rag_app.observability.logger import logger

DB_PATH = project_root / "data" / "rag.db"

def init_db(conn: sqlite3.Connection):
    conn.execute("PRAGMA foreign_keys = ON;")
    
    # Parents table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS parents (
            parent_id TEXT PRIMARY KEY,
            parent_index INTEGER,
            document_id TEXT,
            document_checksum TEXT,
            source_file TEXT,
            source_path TEXT,
            domain TEXT,
            document_type TEXT,
            loader_used TEXT,
            page_number INTEGER,
            character_count INTEGER,
            is_empty_page BOOLEAN,
            ingested_at TEXT,
            chunking_config_hash TEXT,
            page_start INTEGER,
            page_end INTEGER,
            status TEXT,
            version TEXT,
            updated_at TEXT,
            indexed_at TEXT,
            chapter TEXT,
            section TEXT,
            subsection TEXT,
            section_path TEXT,
            parent_checksum TEXT,
            text TEXT
        )
    """)

    # Children table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS children (
            child_id TEXT PRIMARY KEY,
            parent_id TEXT NOT NULL,
            child_index INTEGER,
            document_id TEXT,
            document_checksum TEXT,
            source_file TEXT,
            source_path TEXT,
            domain TEXT,
            document_type TEXT,
            loader_used TEXT,
            page_number INTEGER,
            character_count INTEGER,
            is_empty_page BOOLEAN,
            ingested_at TEXT,
            chunking_config_hash TEXT,
            page_start INTEGER,
            page_end INTEGER,
            status TEXT,
            version TEXT,
            updated_at TEXT,
            indexed_at TEXT,
            chapter TEXT,
            section TEXT,
            subsection TEXT,
            section_path TEXT,
            child_checksum TEXT,
            text TEXT,
            FOREIGN KEY(parent_id) REFERENCES parents(parent_id)
        )
    """)
    conn.commit()

def extract_parent_params(p) -> tuple:
    meta = p.metadata
    return (
        meta.get("parent_id"),
        meta.get("parent_index"),
        meta.get("document_id"),
        meta.get("document_checksum"),
        meta.get("source_file"),
        meta.get("source_path"),
        meta.get("domain"),
        meta.get("document_type"),
        meta.get("loader_used"),
        meta.get("page_number"),
        meta.get("character_count"),
        meta.get("is_empty_page"),
        meta.get("ingested_at"),
        meta.get("chunking_config_hash"),
        meta.get("page_start"),
        meta.get("page_end"),
        meta.get("status"),
        meta.get("version"),
        meta.get("updated_at"),
        meta.get("indexed_at"),
        meta.get("chapter"),
        meta.get("section"),
        meta.get("subsection"),
        meta.get("section_path"),
        meta.get("parent_checksum"),
        p.page_content
    )

def extract_child_params(c) -> tuple:
    meta = c.metadata
    return (
        meta.get("child_id"),
        meta.get("parent_id"),
        meta.get("child_index"),
        meta.get("document_id"),
        meta.get("document_checksum"),
        meta.get("source_file"),
        meta.get("source_path"),
        meta.get("domain"),
        meta.get("document_type"),
        meta.get("loader_used"),
        meta.get("page_number"),
        meta.get("character_count"),
        meta.get("is_empty_page"),
        meta.get("ingested_at"),
        meta.get("chunking_config_hash"),
        meta.get("page_start"),
        meta.get("page_end"),
        meta.get("status"),
        meta.get("version"),
        meta.get("updated_at"),
        meta.get("indexed_at"),
        meta.get("chapter"),
        meta.get("section"),
        meta.get("subsection"),
        meta.get("section_path"),
        meta.get("child_checksum"),
        c.page_content
    )

def ingest_to_sqlite(chunk_groups):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    parents_inserted = 0
    children_inserted = 0
    failures = 0
    
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    
    parent_sql = """
        INSERT OR IGNORE INTO parents (
            parent_id, parent_index, document_id, document_checksum, source_file, source_path,
            domain, document_type, loader_used, page_number, character_count, is_empty_page,
            ingested_at, chunking_config_hash, page_start, page_end, status, version,
            updated_at, indexed_at, chapter, section, subsection, section_path, parent_checksum, text
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    child_sql = """
        INSERT OR IGNORE INTO children (
            child_id, parent_id, child_index, document_id, document_checksum, source_file, source_path,
            domain, document_type, loader_used, page_number, character_count, is_empty_page,
            ingested_at, chunking_config_hash, page_start, page_end, status, version,
            updated_at, indexed_at, chapter, section, subsection, section_path, child_checksum, text
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    for group in chunk_groups:
        parent = group["parent"]
        children = group["children"]
        
        try:
            with conn: # Transaction safety
                conn.execute("PRAGMA foreign_keys = ON;")
                
                cur = conn.cursor()
                cur.execute(parent_sql, extract_parent_params(parent))
                if cur.rowcount > 0:
                    parents_inserted += 1
                
                for child in children:
                    cur.execute(child_sql, extract_child_params(child))
                    if cur.rowcount > 0:
                        children_inserted += 1
                        
        except Exception as e:
            logger.error(f"Failed to persist parent {parent.metadata.get('parent_id')} and children. Rolling back transaction: {e}")
            failures += 1
            
    conn.close()
    return parents_inserted, children_inserted, failures

def run_validation():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    cur.execute("SELECT COUNT(*) FROM parents")
    parent_count = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM children")
    child_count = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM children WHERE parent_id IS NULL")
    null_parent_count = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM children c LEFT JOIN parents p ON c.parent_id = p.parent_id WHERE p.parent_id IS NULL")
    orphan_count = cur.fetchone()[0]
    
    logger.info("=== VALIDATION RESULTS ===")
    logger.info(f"Total Parents: {parent_count}")
    logger.info(f"Total Children: {child_count}")
    logger.info(f"Children with NULL parent_id: {null_parent_count}")
    logger.info(f"Orphaned Children (no matching parent): {orphan_count}")
    
    assert null_parent_count == 0, "Validation failed: Found children with NULL parent_id."
    assert orphan_count == 0, "Validation failed: Found orphaned children."
    logger.info("Validation passed successfully.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true", help="Run database validation after ingestion")
    args = parser.parse_args()

    start_time = time.perf_counter()
    
    logger.info("Starting ingestion script...")
    
    docs, summary = load_documents(
        pdf_dir=project_root / "data" / "pdf",
        metadata_csv_path=project_root / "configs" / "document_metadata.csv"
    )
    
    if not docs:
        logger.warning("No documents loaded.")
        return
        
    chunk_groups = create_parent_child_chunks(docs)
    
    if not chunk_groups:
        logger.warning("No chunks created.")
        return
        
    logger.info(f"Persisting {len(chunk_groups)} parent chunks to SQLite database...")
    parents_inserted, children_inserted, failures = ingest_to_sqlite(chunk_groups)
    
    duration = time.perf_counter() - start_time
    logger.info(
        f"Ingestion completed. Documents received: {summary['documents_loaded']}, "
        f"Parents persisted: {parents_inserted}, Children persisted: {children_inserted}, "
        f"Failures: {failures}. Duration: {duration:.3f}s"
    )

    if args.validate:
        run_validation()

if __name__ == "__main__":
    main()
