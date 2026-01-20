import sqlite3
from src.config.config import DB_PATH, TABLE_NAME

def get_db_connection():
    """Establishes and returns a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    return conn