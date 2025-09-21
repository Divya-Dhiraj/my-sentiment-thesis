# src/pipelines/ingest_real_data.py
import os
import sys
import pandas as pd
import numpy as np

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.database import engine, create_tables

DATA_FILE_PATH = os.path.join(project_root, 'data', 'processed_data.csv')

def run():
    """
    Main function to run the entire data ingestion and processing pipeline.
    This version is fully data-driven and does not contain hard-coded categories.
    """
    print(f"--- 🚀 Starting FULL, Data-Driven Ingestion Pipeline ---")
    create_tables() # Ensure tables exist with the correct schema
    
    try:
        df = pd.read_csv(DATA_FILE_PATH, sep='\t', low_memory=False)
        print(f"✅ Successfully loaded {len(df)} rows from {DATA_FILE_PATH}.")
        
        # --- Data Cleaning & Preparation ---
        df['concession_creation_day'] = pd.to_datetime(df['concession_creation_day'], errors='coerce')
        df.dropna(subset=['concession_creation_day', 'asin'], inplace=True)
        
        # Define all columns that should be treated as strings and fill N/A
        all_str_cols = [
            'gl_name', 'product_type', 'manufacturer_name', 'concession_reason', 
            'defect_category', 'root_cause', 'subcategory_description', 'fulfillment_channel',
            'gl_product_group', 'asp_band', 'clean_category'
        ]
        for col in all_str_cols:
            if col in df.columns:
                df[col] = df[col].fillna('N/A')
        print("✅ Data loaded and pre-cleaned successfully.")

        print("\n--- Starting PostgreSQL Ingestion (Full Schema) ---")
        
        # --- 1. PRODUCTS TABLE (All descriptive columns) ---
        # This list now contains every possible descriptive column from your data.
        product_columns = [
            'asin', 'gl_name', 'product_type', 'clean_category', 'manufacturer_name', 
            'subcategory_code', 'subcategory_description', 'gl_product_group', 'asp_band'
        ]
        # Filter for only columns that actually exist in the CSV to prevent errors
        existing_product_cols = [col for col in product_columns if col in df.columns]
        products_df = df[existing_product_cols].drop_duplicates(subset=['asin']).copy()
        # Rename for consistency with the agent's prompt
        products_df.rename(columns={'gl_name': 'product_name', 'clean_category': 'category'}, inplace=True)
        
        # Use if_exists='replace' to ensure a fresh, clean table every time
        products_df.to_sql('products', con=engine, if_exists='replace', index=False)
        print(f"✅ Loaded {len(products_df)} unique products with full details into 'products' table.")

        # --- 2. CONCESSIONS TABLE (All return/concession-specific columns) ---
        concession_columns = [
            'asin', 'customer_id', 'fulfillment_channel', 'concession_creation_day',
            'concession_reason', 'defect_category', 'root_cause',
            'total_units_conceded', 'our_price', 'marketplace_id'
        ]
        existing_concession_cols = [col for col in concession_columns if col in df.columns]
        concessions_df = df[df['total_units_conceded'] > 0][existing_concession_cols].copy()
        
        concessions_df.to_sql('concessions', con=engine, if_exists='replace', index=False)
        print(f"✅ Loaded {len(concessions_df)} full records into 'concessions' table.")

        # --- 3. WEEKLY PERFORMANCE TABLE (All aggregated financial metrics) ---
        sales_df = df[df['shipped_units'] > 0].copy()
        sales_df['week_start_date'] = pd.to_datetime(sales_df['concession_creation_day']).dt.to_period('W-MON').dt.start_time

        weekly_performance = sales_df.groupby(['week_start_date', 'asin']).agg(
            total_units_sold=('shipped_units', 'sum'),
            product_gms=('product_gms', 'sum'),
            shipped_cogs=('shipped_cogs', 'sum')
        ).reset_index().fillna(0)

        weekly_performance['total_units_sold'] = weekly_performance['total_units_sold'].astype(int)
        
        weekly_performance.to_sql('weekly_performance', con=engine, if_exists='replace', index=False)
        print(f"✅ Loaded {len(weekly_performance)} weekly financial summaries into 'weekly_performance' table.")
        
        print("--- ✅ Finished PostgreSQL Ingestion ---")

    except Exception as e:
        print(f"❌ ERROR during data processing or PostgreSQL ingestion: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n🎉 --- Full Data Ingestion Pipeline Finished Successfully --- 🎉")

if __name__ == "__main__":
    run()