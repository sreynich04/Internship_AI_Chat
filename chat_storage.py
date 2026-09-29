import os
import sqlite3
import libsql
from dotenv import load_dotenv

load_dotenv()

TURSO_URL = os.getenv("TURSO_DATABASE_URL")
TURSO_TOKEN = os.getenv("TURSO_AUTH_TOKEN")


def get_connection():
    """Connects to Turso cloud database with instant local fallback if network lags."""
    if TURSO_URL and TURSO_TOKEN:
        try:
            return libsql.connect(database=TURSO_URL, auth_token=TURSO_TOKEN)
        except Exception as e:
            print(f"⚠️ Turso cloud connection failed: {e}. Falling back to local SQLite.")

    # Local SQLite fallback
    conn = sqlite3.connect("analytics.db", check_same_thread=False, timeout=3)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db():
    """Initializes database tables securely."""
    conn = None
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
        print("✅ Database successfully connected & initialized.")
    except Exception as e:
        print(f"❌ DB Init Warning: {e}")
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def save_to_history_file(session_id, role, content):
    """Saves a conversation turn safely without blocking Flask."""
    conn = None
    try:
        conn = get_connection()
        conn.execute("""
            INSERT INTO chat_history (session_id, role, content)
            VALUES (?, ?, ?)
        """, (str(session_id), role, content))
        conn.commit()
    except Exception as e:
        print(f"⚠️ History save skipped (non-critical): {e}")
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def get_history(session_id):
    """Retrieves conversation history safely."""
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT role, content FROM chat_history
            WHERE session_id = ?
            ORDER BY id ASC
            LIMIT 15
        """, (str(session_id),))
        rows = cursor.fetchall()
        return [{"role": row[0], "content": row[1]} for row in rows]
    except Exception as e:
        print(f"⚠️ History fetch skipped (non-critical): {e}")
        return []
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def log_recommendation(session_id, user_persona, top_major, top_score, mode):
    """Logs recommendation metrics safely."""
    conn = None
    try:
        conn = get_connection()
        conn.execute("""
            INSERT INTO recommendation_logs (session_id, user_persona, top_major, top_score, mode)
            VALUES (?, ?, ?, ?, ?)
        """, (str(session_id), user_persona, top_major, top_score, mode))
        conn.commit()
    except Exception as e:
        print(f"⚠️ Analytics log skipped (non-critical): {e}")
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    init_db()