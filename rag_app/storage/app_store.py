import sqlite3
import json
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "data" / "rag.db"

def init_app_tables():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            history TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS response_cache (
            query TEXT PRIMARY KEY,
            response TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def create_session(session_id: str):
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("INSERT OR IGNORE INTO sessions (session_id, history) VALUES (?, ?)", (session_id, json.dumps([])))
    conn.commit()
    conn.close()

def get_session_history(session_id: str) -> list[dict]:
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute("SELECT history FROM sessions WHERE session_id = ?", (session_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return []

def update_session_history(session_id: str, history: list[dict]):
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("UPDATE sessions SET history = ? WHERE session_id = ?", (json.dumps(history), session_id))
    conn.commit()
    conn.close()

def get_cached_response(query: str) -> dict | None:
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute("SELECT response FROM response_cache WHERE query = ?", (query,))
    row = cur.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return None

def set_cached_response(query: str, response: dict):
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("INSERT OR REPLACE INTO response_cache (query, response) VALUES (?, ?)", (query, json.dumps(response)))
    conn.commit()
    conn.close()
