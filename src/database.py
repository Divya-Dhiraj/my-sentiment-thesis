# src/database.py
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, text

load_dotenv()

def get_db_engine():
    """Establishes and returns a SQLAlchemy engine for the PostgreSQL database."""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL environment variable is not set.")
    
    sqlalchemy_url = db_url.replace("postgresql://", "postgresql+psycopg2://")
    return create_engine(sqlalchemy_url)

# Create a single, reusable engine for the application
engine = get_db_engine()

def create_tables():
    """Connects via the engine and ensures the correct tables exist by dropping old ones first."""
    try:
        with engine.connect() as conn:
            print("Ensuring fresh database tables...")
            
            # --- THE FIX: Drop existing tables to ensure a clean slate ---
            # CASCADE ensures that any dependent objects (like indexes) are also dropped.
            conn.execute(text("DROP TABLE IF EXISTS concessions CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS weekly_performance CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS products CASCADE;"))

            # Now, create the tables with the correct, up-to-date schema
            print("Creating tables with the correct schema...")

            # Table 1: Products
            conn.execute(text("""
            CREATE TABLE products (
                asin VARCHAR(255) PRIMARY KEY,
                product_name TEXT,
                product_type VARCHAR(255)
            );"""))
            
            # Table 2: Weekly Performance
            conn.execute(text("""
            CREATE TABLE weekly_performance (
                record_id SERIAL PRIMARY KEY,
                asin VARCHAR(255) REFERENCES products(asin) ON DELETE CASCADE,
                week_start_date DATE,
                total_units_sold INT,
                average_selling_price FLOAT,
                total_units_conceded INT
            );"""))

            # Table 3: Concessions
            conn.execute(text("""
            CREATE TABLE concessions (
                record_id SERIAL PRIMARY KEY,
                asin VARCHAR(255) REFERENCES products(asin) ON DELETE CASCADE,
                ship_day DATE,
                concession_reason TEXT,
                defect_category TEXT,
                root_cause TEXT
            );"""))

            # Create Indexes for faster queries
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_perf_asin_week ON weekly_performance (asin, week_start_date);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_concessions_asin ON concessions (asin);"))
            
            conn.commit()

            print("✅ Tables and indexes created successfully.")

    except Exception as e:
        print(f"❌ An error occurred during table creation: {e}")