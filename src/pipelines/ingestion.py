# src/pipelines/ingestion.py
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
from pathlib import Path
from ..database import engine # <-- Import the SQLAlchemy engine
from dotenv import load_dotenv
import psycopg2.extras
from sqlalchemy import text


def run():
    """
    Dynamically finds and reads generated CSV data, creates embeddings, 
    and loads everything into the databases using the SQLAlchemy engine.
    """
    load_dotenv()
    print("--- Starting Universal Data Ingestion Pipeline ---")

    try:
        data_dir = Path("/app/data")
        reviews_path = data_dir / "reviews.csv"
        performance_path = data_dir / "weekly_product_performance.csv"

        print(f"Loading reviews from: {reviews_path}")
        reviews_df = pd.read_csv(reviews_path)
        print(f"Loading performance data from: {performance_path}")
        performance_df = pd.read_csv(performance_path)
    except FileNotFoundError as e:
        print(f"❌ ERROR: Data file not found. Make sure you have run the data generator script.")
        print(f"Details: {e}")
        return

    # --- THIS IS THE CORRECTED DATABASE HANDLING LOGIC ---
    pg_conn = None
    try:
        # Get a raw DBAPI connection from the SQLAlchemy engine pool
        pg_conn = engine.raw_connection()
        pg_cursor = pg_conn.cursor()

        # Create tables (this function now uses the engine correctly)
        from .. import database
        database.create_tables()

        print("Clearing existing data from PostgreSQL tables...")
        pg_cursor.execute("TRUNCATE TABLE weekly_performance, reviews, products RESTART IDENTITY CASCADE;")

        chroma_client = chromadb.HttpClient(host=os.environ.get("CHROMA_HOST"), port=int(os.environ.get("CHROMA_PORT")), settings=Settings(anonymized_telemetry=False))
        try:
            chroma_client.delete_collection(name="product_reviews")
            print("Cleared existing data from ChromaDB collection.")
        except Exception:
            print("ChromaDB collection did not exist, creating new one.")

        # The rest of the logic can now proceed as before with the psycopg2 cursor
        products_to_ingest = performance_df[['asin', 'product_name', 'product_type']].drop_duplicates()
        product_tuples = [tuple(x) for x in products_to_ingest.to_numpy()]
        psycopg2.extras.execute_values(pg_cursor,
            "INSERT INTO products (asin, product_name, product_type) VALUES %s", product_tuples)
        print(f"✅ Loaded {len(products_to_ingest)} products into PostgreSQL.")

        perf_cols = ['asin', 'week_start_date', 'average_selling_price', 'discount_percentage', 'total_units_sold', 'num_reviews_received', 'average_rating_new', 'sentiment_score_new']
        performance_tuples = [tuple(x) for x in performance_df[perf_cols].to_numpy()]
        psycopg2.extras.execute_values(pg_cursor,
            "INSERT INTO weekly_performance (asin, week_start_date, average_selling_price, discount_percentage, total_units_sold, num_reviews_received, average_rating_new, sentiment_score_new) VALUES %s", performance_tuples)
        print(f"✅ Loaded {len(performance_df)} weekly performance records into PostgreSQL.")
        
        print("Processing reviews and generating embeddings...")
        embedding_model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')
        
        documents = reviews_df['review_text'].tolist()
        # Ensure documents are strings
        documents = [str(d) for d in documents]
        embeddings = embedding_model.encode(documents, show_progress_bar=True).tolist()
        
        collection = chroma_client.get_or_create_collection(name="product_reviews")
        collection.add(
            ids=reviews_df['review_id'].tolist(),
            embeddings=embeddings,
            documents=documents,
            metadatas=[{'asin': pid, 'rating': int(r)} for pid, r in zip(reviews_df['asin'], reviews_df['rating'])]
        )
        print(f"✅ Loaded {collection.count()} review embeddings into ChromaDB.")
        
        review_cols = ['review_id', 'asin', 'review_date', 'rating', 'review_title', 'review_text']
        review_tuples = [
            tuple(row) + (emb,)
            for (_, row), emb in zip(reviews_df[review_cols].iterrows(), embeddings)
        ]
        psycopg2.extras.execute_values(pg_cursor, 
            "INSERT INTO reviews (review_id, asin, review_date, rating, review_title, review_text, embedding) VALUES %s", review_tuples)
        print(f"✅ Loaded {len(reviews_df)} reviews into PostgreSQL.")

        pg_conn.commit()
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ An error occurred during ingestion: {e}")
        if pg_conn:
            pg_conn.rollback()
    finally:
        if pg_conn:
            pg_cursor.close()
            pg_conn.close()
        print("--- Data Ingestion Pipeline Finished ---")

if __name__ == "__main__":
    run()