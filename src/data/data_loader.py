import pandas as pd
from src.config.config import TABLE_NAME
from src.data.db_connection import get_db_connection
import sqlite3

def load_data():
    conn = get_db_connection()
    query = f"SELECT * FROM {TABLE_NAME}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df