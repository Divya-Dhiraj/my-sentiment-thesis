# src/utils.py
from langchain_community.utilities import SQLDatabase
from .database import engine

def get_db_schema() -> str:
    """Connects to the database and retrieves the schema of all tables."""
    print("--- 📚 Retrieving database schema... ---")
    db = SQLDatabase(engine)
    return db.get_table_info()