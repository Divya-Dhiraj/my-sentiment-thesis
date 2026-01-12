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
    
    # Standardize URL for SQLAlchemy
    if "postgresql://" in db_url:
        sqlalchemy_url = db_url.replace("postgresql://", "postgresql+psycopg2://")
    else:
        sqlalchemy_url = db_url

    # Set pool_pre_ping to True to handle dropped connections in Docker
    return create_engine(sqlalchemy_url, pool_pre_ping=True)

engine = get_db_engine()

def create_tables():
    """Creates the optimized relational schema with explicit lowercase naming."""
    print("--- 🛠️ DEBUG: Building High-Performance Relational Schema ---")
    try:
        with engine.connect() as conn:
            # Drop with CASCADE to ensure a clean slate
            conn.execute(text("DROP TABLE IF EXISTS weekly_performance CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS concessions CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS products CASCADE;"))

            # 1. Products Table
            conn.execute(text("""
                CREATE TABLE products (
                    asin VARCHAR(255) PRIMARY KEY,
                    product_name TEXT,
                    manufacturer_name VARCHAR(255),
                    product_type VARCHAR(255),
                    category VARCHAR(255),
                    asp_band VARCHAR(50),
                    market_segment VARCHAR(255),
                    purchase_intent VARCHAR(255)
                );
            """))
            conn.execute(text("CREATE INDEX idx_products_brand ON products(manufacturer_name);"))

            # 2. Concessions Table
            conn.execute(text("""
                CREATE TABLE concessions (
                    id SERIAL PRIMARY KEY,
                    asin VARCHAR(255) REFERENCES products(asin),
                    customer_id VARCHAR(255),
                    concession_creation_day DATE,
                    concession_reason TEXT,
                    sentiment VARCHAR(50),
                    return_theme VARCHAR(255),
                    suggested_action TEXT,
                    search_vector tsvector GENERATED ALWAYS AS (
                        to_tsvector('english', coalesce(concession_reason, '') || ' ' || coalesce(return_theme, ''))
                    ) STORED
                );
            """))
            conn.execute(text("CREATE INDEX idx_concessions_search ON concessions USING GIN(search_vector);"))

            # 3. Weekly Performance Table
            conn.execute(text("""
                CREATE TABLE weekly_performance (
                    id SERIAL PRIMARY KEY,
                    week_start_date DATE,
                    asin VARCHAR(255) REFERENCES products(asin),
                    total_units_sold INTEGER
                );
            """))
            conn.execute(text("CREATE INDEX idx_weekly_trend ON weekly_performance(week_start_date, asin);"))
            
            conn.commit()
            print("✅ DEBUG: Relational Schema Build Successful.")

    except Exception as e:
        print(f"❌ DEBUG ERROR: {e}")
        raise

if __name__ == "__main__":
    create_tables()