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
    
    # SQLAlchemy requires the format postgresql+psycopg2://...
    sqlalchemy_url = db_url.replace("postgresql://", "postgresql+psycopg2://")
    return create_engine(sqlalchemy_url)

# Create a single, reusable engine for the application
engine = get_db_engine()

def create_tables():
    """Connects via the engine and creates the necessary tables and indexes."""
    try:
        with engine.connect() as conn:
            print("Creating database tables and indexes...")
            
            # Use text() to wrap raw SQL for SQLAlchemy
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS products (
                asin VARCHAR(255) PRIMARY KEY,
                product_name TEXT,
                product_type VARCHAR(255)
            );"""))
            
            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS reviews (
                review_id VARCHAR(255) PRIMARY KEY,
                asin VARCHAR(255) REFERENCES products(asin),
                review_date TIMESTAMP,
                rating INT,
                review_title TEXT,
                review_text TEXT,
                embedding vector(1024)
            );"""))

            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS weekly_performance (
                record_id SERIAL PRIMARY KEY,
                asin VARCHAR(255) REFERENCES products(asin),
                week_start_date DATE,
                average_selling_price FLOAT,
                discount_percentage FLOAT,
                total_units_sold INT,
                num_reviews_received INT,
                average_rating_new FLOAT,
                sentiment_score_new FLOAT
            );"""))

            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_reviews_asin ON reviews (asin);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_reviews_review_date ON reviews (review_date);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_perf_asin ON weekly_performance (asin);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_perf_week_start_date ON weekly_performance (week_start_date);"))
            
            conn.commit() # Commit the transaction

        print("✅ Tables and indexes created successfully.")

    except Exception as e:
        print(f"❌ An error occurred during table creation: {e}")