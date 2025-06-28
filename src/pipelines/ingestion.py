# src/pipelines/ingestion.py    
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
import io
from .. import database

load_dotenv() # This loads variables from the .env file

# --- Add sample financial data ---
financial_data_text = """ProductID,Date,Sales,Price,SupplierCost
A101,2025-03-31,500,899.99,650.00
B202,2025-03-31,1200,549.99,400.00
A101,2025-04-30,850,899.99,650.00
B202,2025-04-30,950,499.99,410.00
"""
financials_df = pd.read_csv(io.StringIO(financial_data_text))
financials_df.to_csv("data/sample_financials.csv", index=False)

# --- Add sample product data ---
product_data_text = """ProductID,ProductName,Category
A101,ProPhone X,Electronics
B202,SmartBook 15,Electronics
"""
products_df = pd.read_csv(io.StringIO(product_data_text))
products_df.to_csv("data/sample_products.csv", index=False)


def run():
    """Reads raw data, generates embeddings, and loads everything into the databases."""
    print("--- Starting Data Ingestion Pipeline ---")
    try:
        # --- 1. Initialize Clients and Models ---
        print("Initializing clients and embedding model...")
        chroma_client = chromadb.HttpClient(host=os.environ.get("CHROMA_HOST", "localhost"), port=int(os.environ.get("CHROMA_PORT", 8000), settings=Settings(anonymized_telemetry=False)))
        embedding_model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')
        pg_conn = database.get_db_connection()
        pg_cursor = pg_conn.cursor()
        print("✅ Clients and model initialized.")

        # --- 2. Create SQL tables ---
        database.create_tables()

        # --- 3. Process and Load Reviews into ChromaDB for RAG ---
        print("Processing reviews for ChromaDB...")
        # Your existing ChromaDB ingestion logic goes here...
        # For now, we'll just confirm the collection exists.
        collection = chroma_client.get_or_create_collection(name="product_reviews")
        print(f"✅ ChromaDB collection 'product_reviews' is ready.")

        # --- 4. Load Products and Financials into PostgreSQL ---
        print("Loading structured data into PostgreSQL...")
        # Load Products
        products_df = pd.read_csv("data/sample_products.csv")
        for _, row in products_df.iterrows():
            pg_cursor.execute("INSERT INTO products (product_id, product_name, category) VALUES (%s, %s, %s) ON CONFLICT (product_id) DO NOTHING;", 
                  (row['ProductID'], row['ProductName'], row['Category']))
        
        # Load Financials
        financials_df = pd.read_csv("data/sample_financials.csv")
        for _, row in financials_df.iterrows():
            pg_cursor.execute("INSERT INTO financials (product_id, date, sales_units, price, supplier_cost) VALUES (%s, %s, %s, %s, %s)",
                            (row['ProductID'], row['Date'], row['Sales'], row['Price'], row['SupplierCost']))
        
        pg_conn.commit()
        print(f"✅ Loaded {len(products_df)} products and {len(financials_df)} financial records into PostgreSQL.")

        pg_cursor.close()
        pg_conn.close()

    except Exception as e:
        print(f"❌ An error occurred during ingestion: {e}")

    print("--- Data Ingestion Pipeline Finished ---")

if __name__ == "__main__":
    run()