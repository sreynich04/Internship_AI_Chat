import sqlite3

def get_connection():
    """Uses standard Python sqlite3 to prevent C-extension socket deadlocks on Render."""
    conn = sqlite3.connect("analytics.db", check_same_thread=False, timeout=5)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db():
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
        print("✅ Local SQLite database initialized successfully.")
    except Exception as e:
        print(f"❌ DB Init Warning: {e}")
    finally:
        if conn:
            conn.close()


def save_to_history_file(session_id, role, content):
    conn = None
    try:
        conn = get_connection()
        conn.execute("""
            INSERT INTO chat_history (session_id, role, content)
            VALUES (?, ?, ?)
        """, (str(session_id), role, content))
        conn.commit()
    except Exception as e:
        print(f"⚠️ History save error: {e}")
    finally:
        if conn:
            conn.close()


def get_history(session_id):
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
        print(f"⚠️ History fetch error: {e}")
        return []
    finally:
        if conn:
            conn.close()


def log_recommendation(session_id, user_persona, top_major, top_score, mode):
    conn = None
    try:
        conn = get_connection()
        conn.execute("""
            INSERT INTO recommendation_logs (session_id, user_persona, top_major, top_score, mode)
            VALUES (?, ?, ?, ?, ?)
        """, (str(session_id), user_persona, top_major, top_score, mode))
        conn.commit()
    except Exception as e:
        print(f"⚠️ Analytics log error: {e}")
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    init_db()