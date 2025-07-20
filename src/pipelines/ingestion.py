# src/pipelines/ingestion.py
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
import io
from .. import database
from dotenv import load_dotenv
import psycopg2.extras # <-- Import for efficient inserts

# --- Realistic Sample Data ---
product_data_text = """ProductID,ProductName,Category
IP16PRO,Apple iPhone 16 Pro,Smartphones
XPS15-9540,Dell XPS 15 (9540),Laptops
"""
review_data_text = """ProductID,ReviewID,ReviewText
IP16PRO,R1,"The new camera system is breathtaking, especially in low light. The battery life, however, is a huge letdown and barely lasts a day."
IP16PRO,R2,"I love the ProMotion display, it's so smooth. But the phone gets surprisingly hot when gaming."
XPS15-9540,R3,"The OLED screen is the best I've ever seen on a laptop. The keyboard is also a dream to type on."
XPS15-9540,R4,"It's powerful, but the fan noise is incredibly loud, even when doing light work. It's very distracting."
IP16PRO,R5,"Finally, USB-C! Transfer speeds are amazing. It feels like a truly 'pro' device now."
"""
financial_data_text = """ProductID,Date,Sales,Price,SupplierCost
IP16PRO,2025-03-31,8000,1199.00,850.00
XPS15-9540,2025-03-31,3500,1899.00,1400.00
IP16PRO,2025-04-30,9500,1199.00,850.00
XPS15-9540,2025-04-30,3200,1849.00,1400.00
"""

def get_sample_data():
    """Creates and returns in-memory DataFrames."""
    products_df = pd.read_csv(io.StringIO(product_data_text))
    reviews_df = pd.read_csv(io.StringIO(review_data_text))
    financials_df = pd.read_csv(io.StringIO(financial_data_text))
    return products_df, reviews_df, financials_df

def run():
    """Reads raw data, generates embeddings, and loads everything into the databases."""
    load_dotenv()
    products_df, reviews_df, financials_df = get_sample_data()
    print("--- Starting Data Ingestion Pipeline with Realistic Data ---")
    
    pg_conn = None
    try:
        # --- 1. SETUP DATABASES ---
        pg_conn = database.get_db_connection()
        pg_cursor = pg_conn.cursor()
        database.create_tables()

        pg_cursor.execute("TRUNCATE TABLE financials, reviews, products RESTART IDENTITY CASCADE;")
        print("Cleared existing data from PostgreSQL tables.")

        chroma_client = chromadb.HttpClient(host=os.environ.get("CHROMA_HOST"), port=int(os.environ.get("CHROMA_PORT")), settings=Settings(anonymized_telemetry=False))
        try:
            chroma_client.delete_collection(name="product_reviews")
            print("Cleared existing data from ChromaDB collection.")
        except:
            print("ChromaDB collection did not exist, creating new one.")

        # --- 2. INGEST PRODUCTS AND FINANCIALS INTO POSTGRESQL ---
        print("Loading structured data into PostgreSQL...")
        
        # More efficient bulk insert for products
        product_tuples = [tuple(x) for x in products_df.to_numpy()]
        psycopg2.extras.execute_values(pg_cursor, 
            "INSERT INTO products (product_id, product_name, category) VALUES %s", product_tuples)

        # More efficient bulk insert for financials
        financial_tuples = [tuple(x) for x in financials_df.to_numpy()]
        psycopg2.extras.execute_values(pg_cursor, 
            "INSERT INTO financials (product_id, date, sales_units, price, supplier_cost) VALUES %s", financial_tuples)
        
        print(f"✅ Loaded {len(products_df)} products and {len(financials_df)} financial records into PostgreSQL.")

        # --- 3. INGEST REVIEWS INTO BOTH DATABASES ---
        print("Processing and loading reviews into ChromaDB (for RAG) and PostgreSQL (for records)...")
        
        # A. Load into ChromaDB with embeddings
        embedding_model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')
        collection = chroma_client.get_or_create_collection(name="product_reviews")
        
        documents = reviews_df['ReviewText'].tolist()
        embeddings = embedding_model.encode(documents, show_progress_bar=True).tolist()
        
        collection.add(
            ids=[str(id) for id in reviews_df['ReviewID']],
            embeddings=embeddings,
            documents=documents,
            metadatas=[{'product_id': pid} for pid in reviews_df['ProductID']]
        )
        print(f"✅ Loaded {collection.count()} review embeddings into ChromaDB.")
        
        # B. Load into PostgreSQL with embeddings
        review_tuples = [
            (row['ReviewID'], row['ProductID'], row['ReviewText'], emb)
            for (_, row), emb in zip(reviews_df.iterrows(), embeddings)
        ]
        psycopg2.extras.execute_values(pg_cursor, 
            "INSERT INTO reviews (review_id, product_id, review_text, embedding) VALUES %s", review_tuples)
        print(f"✅ Loaded {len(reviews_df)} reviews into PostgreSQL.")

        # --- 4. FINALIZE ---
        pg_conn.commit()
        
    except Exception as e:
        print(f"❌ An error occurred during ingestion: {e}")
        if pg_conn:
            pg_conn.rollback() # Rollback changes if an error occurs
    finally:
        if pg_conn:
            pg_cursor.close()
            pg_conn.close()
        print("--- Data Ingestion Pipeline Finished ---")

if __name__ == "__main__":
    run()