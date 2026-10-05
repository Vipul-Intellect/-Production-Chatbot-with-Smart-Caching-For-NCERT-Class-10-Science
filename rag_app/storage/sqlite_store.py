import sqlite3
from typing import Any
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "data" / "rag.db"

def get_sqlite_connection() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"SQLite database not found at {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def fetch_children_by_ids(child_ids: list[str]) -> list[dict[str, Any]]:
    if not child_ids:
        return []
    
    conn = get_sqlite_connection()
    try:
        placeholders = ",".join("?" for _ in child_ids)
        # We strictly enforce only retrieving ACTIVE chunks for production search
        query = f"SELECT * FROM children WHERE child_id IN ({placeholders}) AND status = 'ACTIVE'"
        cur = conn.cursor()
        cur.execute(query, child_ids)
        rows = cur.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()

def fetch_parents_by_ids(parent_ids: list[str]) -> list[dict[str, Any]]:
    if not parent_ids:
        return []
    
    conn = get_sqlite_connection()
    try:
        placeholders = ",".join("?" for _ in parent_ids)
        # Only ACTIVE chunks
        query = f"SELECT * FROM parents WHERE parent_id IN ({placeholders}) AND status = 'ACTIVE'"
        cur = conn.cursor()
        cur.execute(query, parent_ids)
        rows = cur.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()
