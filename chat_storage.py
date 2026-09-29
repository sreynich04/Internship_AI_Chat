import os
import sqlite3
import requests

TURSO_DATABASE_URL = os.getenv("TURSO_DATABASE_URL", "")
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN", "")

def get_turso_endpoint():
    if not TURSO_DATABASE_URL:
        return None
    url = TURSO_DATABASE_URL.replace("libsql://", "https://").replace("turso://", "https://")
    if not url.startswith("http"):
        url = f"https://{url}"
    return url.rstrip("/") + "/v2/pipeline"

def query_turso_http(sql, args=None):
    """
    Executes SQL against Turso over HTTPS with a strict 3-second socket timeout.
    Prevents C-library native socket hangs in Gunicorn workers.
    """
    endpoint = get_turso_endpoint()
    if not endpoint or not TURSO_AUTH_TOKEN:
        return None

    formatted_args = []
    if args:
        for arg in args:
            if isinstance(arg, int):
                formatted_args.append({"type": "integer", "value": str(arg)})
            else:
                formatted_args.append({"type": "text", "value": str(arg)})

    stmt = {"sql": sql}
    if formatted_args:
        stmt["args"] = formatted_args

    payload = {
        "requests": [
            {"type": "execute", "stmt": stmt},
            {"type": "close"}
        ]
    }
    headers = {
        "Authorization": f"Bearer {TURSO_AUTH_TOKEN}",
        "Content-Type": "application/json"
    }

    response = requests.post(endpoint, json=payload, headers=headers, timeout=3.0)
    response.raise_for_status()
    data = response.json()

    result = data["results"][0]["response"]["result"]
    raw_rows = result.get("rows", [])
    clean_rows = []
    for row in raw_rows:
        clean_rows.append([col.get("value") for col in row])
    return clean_rows

# --- Local SQLite Fallback Helpers ---

def query_sqlite(sql, args=(), fetch=False):
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute(sql, args)
    rows = cursor.fetchall() if fetch else None
    conn.commit()
    conn.close()
    return rows

# --- Public Functions ---

def init_db():
    queries = [
        """CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS recommendation_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            recommendation TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )"""
    ]
    
    for q in queries:
        try:
            query_sqlite(q)
        except Exception as e:
            print(f"❌ SQLite init error: {e}")

    for q in queries:
        try:
            query_turso_http(q)
            print("⚡ Turso Database verified via HTTP!")
        except Exception as e:
            print(f"⚠️ Turso HTTP init skipped/failed: {e}")
            break

def save_message(session_id, role, content):
    sql = "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)"
    args = (session_id, role, content)

    try:
        query_turso_http(sql, args)
        print(f"⚡ Saved message for {session_id} to Turso Cloud!")
        return
    except Exception as e:
        print(f"⚠️ Turso HTTP save failed ({e}). Falling back to local SQLite.")

    try:
        query_sqlite(sql, args)
        print(f"✅ Saved message for {session_id} to local SQLite.")
    except Exception as e:
        print(f"❌ save_message local SQLite error: {e}")

def save_to_history_file(session_id, role, content):
    save_message(session_id, role, content)

def get_history(session_id, limit=10):
    sql = "SELECT role, content FROM chat_history WHERE session_id = ? ORDER BY id DESC LIMIT ?"
    args = (session_id, limit)

    try:
        rows = query_turso_http(sql, args)
        if rows is not None:
            return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
    except Exception as e:
        print(f"⚠️ Turso HTTP get_history failed ({e}). Falling back to local SQLite.")

    try:
        rows = query_sqlite(sql, args, fetch=True)
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
    except Exception as e:
        print(f"❌ get_history error: {e}")
        return []

def log_recommendation(*args, **kwargs):
    session_id = str(args[0]) if len(args) > 0 else str(kwargs.get("session_id", "unknown"))
    recommendation = ", ".join(map(str, args[1:])) if len(args) > 1 else str(kwargs)

    sql = "INSERT INTO recommendation_logs (session_id, recommendation) VALUES (?, ?)"
    sql_args = (session_id, recommendation)

    try:
        query_turso_http(sql, sql_args)
        print(f"⚡ Logged recommendation for {session_id} to Turso Cloud!")
        return
    except Exception as e:
        print(f"⚠️ Turso HTTP log failed ({e}). Falling back to local SQLite.")

    try:
        query_sqlite(sql, sql_args)
        print(f"✅ Logged recommendation for {session_id} to local SQLite.")
    except Exception as e:
        print(f"❌ log_recommendation error: {e}")