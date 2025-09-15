# src/pipelines/ingest_real_data.py
import os
import sys
import pandas as pd
from sqlalchemy import text
import chromadb
from sentence_transformers import SentenceTransformer

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.database import engine, create_tables

DATA_FILE_PATH = os.path.join(project_root, 'data', 'processed_data.csv')

def run():
    """Main function to run the ETL pipeline for both PostgreSQL and ChromaDB."""
    print("--- 🚀 Starting Full Data Ingestion Pipeline (SQL + Vector) ---")
    
    # This function now drops and recreates all tables to ensure a clean slate
    create_tables()

    try:
        df = pd.read_csv(DATA_FILE_PATH, sep='\t', low_memory=False)
        # Coerce errors will turn unparseable dates into NaT (Not a Time)
        df['ship_day'] = pd.to_datetime(df['ship_day'], errors='coerce')
        # Drop rows where the date could not be parsed
        df.dropna(subset=['ship_day'], inplace=True)
        print("✅ Data loaded and dates parsed successfully.")
    except FileNotFoundError:
        print(f"❌ ERROR: Data file not found at {DATA_FILE_PATH}.")
        return

    # --- 1. PostgreSQL Ingestion ---
    print("\n--- Starting PostgreSQL Ingestion ---")
    try:
        # === Ingest Products (Parent Table) ===
        products_df = df[['asin', 'gl_name', 'product_type']].drop_duplicates(subset=['asin']).copy()
        products_df.rename(columns={'gl_name': 'product_name'}, inplace=True)
        products_df.dropna(subset=['asin'], inplace=True)
        products_df.to_sql('products', con=engine, if_exists='append', index=False)
        # THIS IS THE CORRECTED LINE:
        print(f"✅ Loaded {len(products_df)} unique products into 'products' table.")

        # === Prepare and Ingest Concessions Data ===
        concessions_df = df[df['concession_reason'].notnull()].copy()
        concessions_columns = ['asin', 'ship_day', 'concession_reason', 'defect_category', 'root_cause']
        concessions_df[concessions_columns].to_sql('concessions', con=engine, if_exists='append', index=False)
        print(f"✅ Loaded {len(concessions_df)} records into 'concessions' table.")

        # === Prepare and Ingest Weekly Performance Data ===
        # Use a copy to avoid SettingWithCopyWarning
        orders_df = df[df['concession_reason'].isnull()].copy()

        # Aggregate sales from orders
        weekly_sales = orders_df.groupby(
            [pd.Grouper(key='ship_day', freq='W-MON'), 'asin']
        ).agg(
            total_units_sold=('shipped_units', 'sum'),
            average_selling_price=('our_price', 'mean')
        ).reset_index()

        # Aggregate returns from concessions data
        weekly_returns = concessions_df.groupby(
            [pd.Grouper(key='ship_day', freq='W-MON'), 'asin']
        ).agg(
            total_units_conceded=('total_units_conceded', 'sum')
        ).reset_index()

        # Merge sales and returns data
        weekly_performance_df = pd.merge(
            weekly_sales,
            weekly_returns,
            on=['ship_day', 'asin'],
            how='outer'
        ).fillna(0)

        # Rename column to match the database schema
        weekly_performance_df.rename(columns={'ship_day': 'week_start_date'}, inplace=True)
        
        # Ingest into the weekly_performance table
        performance_columns = ['week_start_date', 'asin', 'total_units_sold', 'average_selling_price', 'total_units_conceded']
        weekly_performance_df[performance_columns].to_sql('weekly_performance', con=engine, if_exists='append', index=False)
        print(f"✅ Loaded {len(weekly_performance_df)} weekly summary records into 'weekly_performance' table.")
        print("--- ✅ Finished PostgreSQL Ingestion ---")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ ERROR during PostgreSQL ingestion: {e}")
        return

    # --- 2. ChromaDB Vector Ingestion ---
    print("\n--- Starting ChromaDB Vector Ingestion ---")
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        products_to_embed = products_df.dropna(subset=['product_name', 'product_type'])
        documents = [f"Product Name: {name}. Type: {ptype}." for name, ptype in zip(products_to_embed['product_name'], products_to_embed['product_type'])]
        asins = products_to_embed['asin'].tolist()

        embeddings = embedding_model.encode(documents, show_progress_bar=True).tolist()
        
        chroma_client = chromadb.HttpClient(host="chroma", port=8000)
        collection_name = "products"
        
        try:
            chroma_client.delete_collection(name=collection_name)
            print(f"Deleted old ChromaDB collection: '{collection_name}'")
        except Exception:
            print(f"ChromaDB collection '{collection_name}' did not exist, creating new.")
        
        collection = chroma_client.create_collection(name=collection_name)
        
        batch_size = 4000
        for i in range(0, len(asins), batch_size):
            batch_ids = asins[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            batch_metadatas = [{"description": doc} for doc in documents[i:i+batch_size]]
            
            print(f"  - Adding batch {i//batch_size + 1} to ChromaDB...")
            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )
        print("--- ✅ Finished ChromaDB Vector Ingestion ---")

    except Exception as e:
        print(f"❌ ERROR during ChromaDB ingestion: {e}")
        return

    print("\n🎉 --- Full Data Ingestion Pipeline Finished Successfully --- 🎉")

if __name__ == "__main__":
    run()