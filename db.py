import sqlite3
from typing import List, Dict

DB_NAME = "chat_history.db"

def init_db():
    """
    Initializes the SQLite database and creates the chat_history table if it doesn't exist.
    """
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

def save_message(role: str, content: str):
    """
    Persists a single message (user or model) into the database.
    """
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_history (role, content) VALUES (?, ?)",
            (role, content)
        )
        conn.commit()

def load_recent_history(limit: int = 10) -> List[Dict]:
    """
    Loads the most recent 'limit' messages formatted for the Gemini API contents payload.
    """
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        # Fetch last N records ordered by ID descending, then reverse to maintain timeline
        cursor.execute("""
            SELECT role, content FROM (
                SELECT id, role, content FROM chat_history ORDER BY id DESC LIMIT ?
            ) ORDER BY id ASC
        """, (limit,))
        
        rows = cursor.fetchall()

    history = []
    for role, content in rows:
        history.append({
            "role": role,
            "parts": [{"text": content}]
        })
    return history

def clear_history():
    """
    Utility to purge all stored sessions.
    """
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_history")
        conn.commit()