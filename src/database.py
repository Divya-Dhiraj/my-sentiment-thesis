# src/database.py
from dotenv import load_dotenv
import psycopg2
import os

load_dotenv() # This loads variables from the .env file
def get_db_connection():
    """Establishes and returns a connection to the PostgreSQL database."""
    # This gets the database URL from the environment variables set in docker-compose
    db_url = os.environ.get("DATABASE_URL")
    return psycopg2.connect(db_url)

def create_tables():
    """Connects to PostgreSQL and creates the necessary tables for the thesis project."""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        print("Creating database tables...")
        
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Table for product specifications
        cur.execute("""
        CREATE TABLE IF NOT EXISTS products (
            product_id VARCHAR(255) PRIMARY KEY,
            product_name TEXT,
            category VARCHAR(255)
        );""")

        # Table for customer reviews
        cur.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            review_id VARCHAR(255) PRIMARY KEY,
            product_id VARCHAR(255) REFERENCES products(product_id),
            review_text TEXT,
            embedding vector(1024)
        );""")

        # --- ADD THIS NEW BLOCK ---
        # Table for financial data
        cur.execute("""
        CREATE TABLE IF NOT EXISTS financials (
            record_id SERIAL PRIMARY KEY,
            product_id VARCHAR(255) REFERENCES products(product_id),
            date DATE,
            price FLOAT,
            sales_units INT,
            supplier_cost FLOAT
        );
        """)
        # --- END OF NEW BLOCK ---
        
        conn.commit()
        cur.close()
        print("✅ Tables 'products', 'reviews', and 'financials' created successfully in PostgreSQL.")

    except Exception as e:
        print(f"❌ An error occurred during table creation: {e}")
    finally:
        if conn is not None:
            conn.close()