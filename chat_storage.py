import os
import sqlite3
import libsql
from dotenv import load_dotenv

load_dotenv()

TURSO_URL = os.getenv("TURSO_DATABASE_URL")
TURSO_TOKEN = os.getenv("TURSO_AUTH_TOKEN")

def get_connection():
    """Connects to Turso cloud database or falls back to local SQLite."""
    if TURSO_URL and TURSO_TOKEN:
        return libsql.connect(database=TURSO_URL, auth_token=TURSO_TOKEN)
    else:
        conn = sqlite3.connect("analytics.db")
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

def init_db():
    """Initializes the recommendation logs and chat history schemas."""
    try:
        conn = get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS recommendation_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT,
                user_persona TEXT,
                top_major TEXT,
                top_score REAL,
                mode TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
        print("✅ Database successfully connected & initialized in Turso Cloud.")
    except Exception as e:
        print(f"❌ Database Initialization Error: {e}")

def log_recommendation(session_id, user_persona, top_major, top_score, mode):
    """Logs recommendation metrics to the central database."""
    try:
        conn = get_connection()
        conn.execute("""
            INSERT INTO recommendation_logs (session_id, user_persona, top_major, top_score, mode)
            VALUES (?, ?, ?, ?, ?)
        """, (str(session_id), user_persona, top_major, top_score, mode))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error logging recommendation: {e}")

def save_to_history_file(session_id, role, content):
    """Saves a conversation turn to chat history in Turso."""
    try:
        conn = get_connection()
        conn.execute("""
            INSERT INTO chat_history (session_id, role, content)
            VALUES (?, ?, ?)
        """, (str(session_id), role, content))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving chat history: {e}")

def get_history(session_id):
    """Retrieves stored conversation history for a specific session."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT role, content FROM chat_history
            WHERE session_id = ?
            ORDER BY id ASC
        """, (str(session_id),))
        rows = cursor.fetchall()
        conn.close()
        return [{"role": row[0], "content": row[1]} for row in rows]
    except Exception as e:
        print(f"Error retrieving history: {e}")
        return []

if __name__ == "__main__":
    init_db()