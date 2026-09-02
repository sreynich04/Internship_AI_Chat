import sqlite3

DB_PATH = "chat_history.db"

def inspect_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("\n================== 1. RECENT CHAT MESSAGES ==================")
    try:
        messages = cursor.execute("""
            SELECT id, session_id, role, content, created_at 
            FROM messages 
            ORDER BY id DESC LIMIT 10
        """).fetchall()
        
        if not messages:
            print("No messages stored yet.")
        else:
            for msg in reversed(messages):
                print(f"[{msg[4]}] ({msg[1]}) {msg[2].upper()}: {msg[3][:80]}...")
    except Exception as e:
        print(f"Error fetching messages: {e}")

    print("\n================= 2. RECOMMENDATION LOGS =================")
    try:
        logs = cursor.execute("""
            SELECT id, session_id, recommended_major, match_score, mode, created_at 
            FROM recommendation_logs 
            ORDER BY id DESC LIMIT 10
        """).fetchall()

        if not logs:
            print("No recommendation logs recorded yet.")
        else:
            for log in reversed(logs):
                print(f"[{log[5]}] Session: {log[1]} | Major: {log[2]} | Score: {log[3]}% | Mode: {log[4]}")
    except Exception as e:
        print(f"Error fetching logs: {e}")

    conn.close()

if __name__ == "__main__":
    inspect_db()