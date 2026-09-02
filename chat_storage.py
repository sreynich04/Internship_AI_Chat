import sqlite3

DB_PATH = "chat_history.db"

def init_db():
    """Initializes message history, recommendation logs, and feedback tables."""
    with sqlite3.connect(DB_PATH) as conn:
        # Chat History Table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Recommendation Analytics Table (For Defense Data & Metrics)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS recommendation_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_persona TEXT NOT NULL,
                recommended_major TEXT NOT NULL,
                match_score REAL NOT NULL,
                mode TEXT NOT NULL, -- 'DISCOVERY' or 'RECOMMENDATION'
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

def save_to_history_file(session_id: str, user_message: str, ai_response: str):
    """Saves user message and AI response."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, "user", user_message)
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, "assistant", ai_response)
        )

def log_recommendation(session_id: str, persona: str, major: str, score: float, mode: str):
    """Logs recommendation decisions for model evaluation and dashboard reporting."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO recommendation_logs (session_id, user_persona, recommended_major, match_score, mode)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, persona, major, score, mode))

def get_history(session_id: str, limit: int = 20) -> list:
    """Loads recent messages in chronological order."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        rows = cursor.execute("""
            SELECT role, content FROM (
                SELECT id, role, content FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
            ) ORDER BY id ASC
        """, (session_id, limit)).fetchall()
        
        return [{"role": r[0], "content": r[1]} for r in rows]