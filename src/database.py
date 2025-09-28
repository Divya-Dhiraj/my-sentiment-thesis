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

engine = get_db_engine()

def create_tables():
    """
    Drops existing tables and recreates them with the final, ENRICHED schema.
    This prepares the database to receive all the newly generated columns.
    """
    print("Ensuring a fresh database with the complete enriched schema...")
    try:
        with engine.connect() as conn:
            print("  - Dropping existing tables (if they exist)...")
            conn.execute(text("DROP TABLE IF EXISTS weekly_performance CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS concessions CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS products CASCADE;"))

            print("  - Creating new tables with enriched columns...")
            
            # 1. Products table with all enriched fields
            conn.execute(text("""
                CREATE TABLE products (
                    asin VARCHAR(255) PRIMARY KEY,
                    product_name TEXT,
                    product_type VARCHAR(255),
                    category VARCHAR(255),
                    manufacturer_name VARCHAR(255),
                    subcategory_code VARCHAR(255),
                    subcategory_description TEXT,
                    asp_band VARCHAR(50),
                    "PurchaseIntent" VARCHAR(255),
                    "MarketSegment" VARCHAR(255),
                    "PredictedDiscountImpact" VARCHAR(50)
                );
            """))

            # 2. Concessions table with all enriched fields
            conn.execute(text("""
                CREATE TABLE concessions (
                    id SERIAL PRIMARY KEY,
                    asin VARCHAR(255) REFERENCES products(asin),
                    customer_id VARCHAR(255),
                    concession_creation_day DATE,
                    concession_reason TEXT,
                    "Sentiment" VARCHAR(50),
                    "ReturnTheme" VARCHAR(255),
                    "SuggestedAction" TEXT
                );
            """))

            # 3. Weekly Performance table (schema is stable)
            conn.execute(text("""
                CREATE TABLE weekly_performance (
                    id SERIAL PRIMARY KEY,
                    week_start_date DATE,
                    asin VARCHAR(255) REFERENCES products(asin),
                    total_units_sold INTEGER
                );
            """))
            
            conn.commit()
            print("✅ Enriched tables and constraints created successfully.")

    except Exception as e:
        print(f"❌ An error occurred during table creation: {e}")
        import traceback
        traceback.print_exc()
        raise