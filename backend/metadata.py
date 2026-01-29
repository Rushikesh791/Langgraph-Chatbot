import sqlite3

DB_PATH = "chatbot.db"

# --------------------------------------------------
# Init metadata table
# --------------------------------------------------
def init_metadata_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS thread_metadata (
            thread_id TEXT PRIMARY KEY,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


# --------------------------------------------------
# Save / Update title
# --------------------------------------------------
def save_thread_title(thread_id: str, title: str):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO thread_metadata (thread_id, title)
        VALUES (?, ?)
    """, (thread_id, title))

    conn.commit()
    conn.close()


# --------------------------------------------------
# Load all titles
# --------------------------------------------------
def load_all_thread_titles() -> dict:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()

    cursor.execute("SELECT thread_id, title FROM thread_metadata")
    rows = cursor.fetchall()

    conn.close()

    return {thread_id: title for thread_id, title in rows}

