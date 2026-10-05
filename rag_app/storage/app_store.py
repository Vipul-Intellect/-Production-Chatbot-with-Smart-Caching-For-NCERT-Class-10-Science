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

# --- Redis Cache Implementation ---
import os
from rag_app.observability.logger import logger
import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_redis_client = None

def get_redis_client():
    global _redis_client
    if _redis_client is None:
        try:
            client = redis.from_url(REDIS_URL, decode_responses=True)
            client.ping()
            _redis_client = client
        except Exception as e:
            logger.warning(f"Redis unavailable: {e}. Semantic cache will be bypassed.")
            _redis_client = False # Disable it
    return _redis_client if _redis_client else None

def get_cached_response(cache_id: str) -> dict | None:
    client = get_redis_client()
    if client:
        try:
            data = client.get(f"cache:{cache_id}")
            if data:
                return json.loads(data)
        except Exception as e:
            logger.warning(f"Failed to read from Redis: {e}")
            
    # Fallback to SQLite
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cur = conn.cursor()
        # Ensure table exists
        conn.execute("CREATE TABLE IF NOT EXISTS fallback_cache (cache_id TEXT PRIMARY KEY, response TEXT)")
        cur.execute("SELECT response FROM fallback_cache WHERE cache_id = ?", (cache_id,))
        row = cur.fetchone()
        conn.close()
        if row:
            return json.loads(row[0])
    except Exception as e:
        logger.warning(f"SQLite cache read failed: {e}")
        
    return None

def set_cached_response(cache_id: str, response: dict):
    client = get_redis_client()
    if client:
        try:
            client.set(f"cache:{cache_id}", json.dumps(response))
            return
        except Exception as e:
            logger.warning(f"Failed to write to Redis: {e}")
            
    # Fallback to SQLite
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("CREATE TABLE IF NOT EXISTS fallback_cache (cache_id TEXT PRIMARY KEY, response TEXT)")
        conn.execute("INSERT OR REPLACE INTO fallback_cache (cache_id, response) VALUES (?, ?)", (cache_id, json.dumps(response)))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"SQLite cache write failed: {e}")

def get_cached_question(cache_id: str) -> str | None:
    client = get_redis_client()
    if client:
        try:
            return client.get(f"cache_q:{cache_id}")
        except Exception:
            pass
            
    # Fallback to SQLite
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cur = conn.cursor()
        conn.execute("CREATE TABLE IF NOT EXISTS fallback_cache_questions (cache_id TEXT PRIMARY KEY, question TEXT)")
        cur.execute("SELECT question FROM fallback_cache_questions WHERE cache_id = ?", (cache_id,))
        row = cur.fetchone()
        conn.close()
        if row:
            return row[0]
    except Exception:
        pass
    return None

def set_cached_question(cache_id: str, question: str):
    client = get_redis_client()
    if client:
        try:
            client.set(f"cache_q:{cache_id}", question)
            return
        except Exception:
            pass
            
    # Fallback to SQLite
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("CREATE TABLE IF NOT EXISTS fallback_cache_questions (cache_id TEXT PRIMARY KEY, question TEXT)")
        conn.execute("INSERT OR REPLACE INTO fallback_cache_questions (cache_id, question) VALUES (?, ?)", (cache_id, question))
        conn.commit()
        conn.close()
    except Exception:
        pass
