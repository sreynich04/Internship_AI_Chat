import os
import sqlite3
import concurrent.futures

TURSO_DATABASE_URL = os.getenv("TURSO_DATABASE_URL")
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN")

def _connect_turso():
    import libsql
    return libsql.connect(database=TURSO_DATABASE_URL, auth_token=TURSO_AUTH_TOKEN)

def get_db_connection():
    if TURSO_DATABASE_URL and TURSO_AUTH_TOKEN:
        try:
            # Strict 3-second timeout guard to prevent Gunicorn worker freezes
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_connect_turso)
                conn = future.result(timeout=3.0)
                print("⚡ Successfully connected to Turso Database!")
                return conn
        except concurrent.futures.TimeoutError:
            print("⏱️ TURSO CONNECTION TIMED OUT (3s limit). Falling back to local SQLite.")
        except Exception as e:
            print(f"❌ TURSO CONNECTION FAILED: {e}. Falling back to local SQLite.")
    else:
        print("⚠️ TURSO_DATABASE_URL or TURSO_AUTH_TOKEN missing. Using local SQLite.")
    
    return sqlite3.connect("chat_history.db")

def init_db():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recommendation_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                recommendation TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"❌ init_db error: {e}")

def save_message(session_id, role, content):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)
        )
        conn.commit()
        conn.close()
        print(f"✅ Saved message for {session_id} to database.")
    except Exception as e:
        print(f"❌ save_message error: {e}")

def save_to_history_file(session_id, role, content):
    save_message(session_id, role, content)

def get_history(session_id, limit=10):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content FROM chat_history WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit)
        )
        rows = cursor.fetchall()
        conn.close()
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
    except Exception as e:
        print(f"❌ get_history error: {e}")
        return []

def log_recommendation(*args, **kwargs):
    try:
        session_id = str(args[0]) if len(args) > 0 else str(kwargs.get("session_id", "unknown"))
        recommendation = ", ".join(map(str, args[1:])) if len(args) > 1 else str(kwargs)

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO recommendation_logs (session_id, recommendation) VALUES (?, ?)",
            (session_id, recommendation)
        )
        conn.commit()
        conn.close()
        print(f"✅ Logged recommendation for {session_id}.")
    except Exception as e:
        print(f"❌ log_recommendation error: {e}")