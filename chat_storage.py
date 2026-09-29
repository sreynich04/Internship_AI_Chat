import os
import sqlite3

TURSO_DATABASE_URL = os.getenv("TURSO_DATABASE_URL")
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN")

def get_db_connection():
    if TURSO_DATABASE_URL and TURSO_AUTH_TOKEN:
        try:
            # Ensure URL uses https:// for the Python libsql client
            url = TURSO_DATABASE_URL.replace("libsql://", "https://")
            
            import libsql
            conn = libsql.connect(database=url, auth_token=TURSO_AUTH_TOKEN)
            print("⚡ Successfully connected to Turso Database!")
            return conn
        except Exception as e:
            print(f"❌ TURSO CONNECTION FAILED: {e}")
            print("⚠️ Falling back to local SQLite.")
    else:
        print("⚠️ TURSO_DATABASE_URL or TURSO_AUTH_TOKEN missing in Environment. Using local SQLite.")
    
    return sqlite3.connect("chat_history.db")

def init_db():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create chat_history table to match your Turso database schema
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
                session_id TEXT NOT NULL,
                recommendation TEXT NOT NULL,
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