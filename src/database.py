# src/database.py
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

def get_db_engine():
    """Establishes and returns a SQLAlchemy engine for the PostgreSQL database."""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL environment variable is not set.")
    
    if "postgresql://" in db_url:
        sqlalchemy_url = db_url.replace("postgresql://", "postgresql+psycopg2://")
    else:
        sqlalchemy_url = db_url

    return create_engine(sqlalchemy_url)

# Create a single, reusable engine for the application
engine = get_db_engine()

def create_tables():
    """
    Drops existing tables in the correct dependency order (children first),
    then recreates them with the complete, final schema. This ensures a clean
    slate for every data ingestion run.
    """
    print("Ensuring a fresh database with the complete schema...")
    try:
        with engine.connect() as conn:
            # Drop tables in reverse order of dependency to avoid foreign key errors
            print("  - Dropping existing tables (if they exist)...")
            conn.execute(text("DROP TABLE IF EXISTS weekly_performance CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS concessions CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS products CASCADE;"))

            # Create tables with the full, comprehensive schema
            print("  - Creating new tables...")

            # 1. Products table (no dependencies)
            conn.execute(text("""
                CREATE TABLE products (
                    asin VARCHAR(255) PRIMARY KEY,
                    product_name TEXT,
                    product_type VARCHAR(255),
                    category VARCHAR(255),
                    manufacturer_name VARCHAR(255),
                    subcategory_code VARCHAR(255),
                    subcategory_description TEXT,
                    gl_product_group VARCHAR(255),
                    asp_band VARCHAR(50)
                );
            """))

            # 2. Concessions table (depends on products)
            conn.execute(text("""
                CREATE TABLE concessions (
                    id SERIAL PRIMARY KEY,
                    asin VARCHAR(255) REFERENCES products(asin),
                    customer_id VARCHAR(255),
                    fulfillment_channel VARCHAR(50),
                    concession_creation_day DATE,
                    concession_reason TEXT,
                    defect_category TEXT,
                    root_cause TEXT,
                    total_units_conceded INTEGER,
                    our_price NUMERIC(10, 2),
                    marketplace_id INTEGER
                );
            """))

            # 3. Weekly Performance table (depends on products)
            conn.execute(text("""
                CREATE TABLE weekly_performance (
                    id SERIAL PRIMARY KEY,
                    week_start_date DATE,
                    asin VARCHAR(255) REFERENCES products(asin),
                    total_units_sold INTEGER,
                    product_gms NUMERIC(12, 2),
                    shipped_cogs NUMERIC(12, 2),
                    UNIQUE(week_start_date, asin)
                );
            """))
            
            # Create Indexes for faster queries
            print("  - Creating indexes for faster queries...")
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_products_category ON products (category);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_perf_asin_week ON weekly_performance (asin, week_start_date);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_concessions_asin ON concessions (asin);"))
            
            conn.commit()
            print("✅ Tables and indexes created successfully.")

    except Exception as e:
        print(f"❌ An error occurred during table creation: {e}")
        import traceback
        traceback.print_exc()
        raise