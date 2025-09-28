# src/pipelines/ingest_real_data.py
import os
import sys
import pandas as pd

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.database import engine, create_tables

# The pipeline now reads from your new, enriched file.
DATA_FILE_PATH = os.path.join(project_root, 'data', 'enriched_data.csv')

def run():
    """
    Main function to load the final, ENRICHED dataset into the database.
    """
    print(f"--- 🚀 Starting Final Ingestion Pipeline with Enriched Data ---")
    create_tables() # This will wipe and recreate the tables with the new, enriched schema
    
    try:
        df = pd.read_csv(DATA_FILE_PATH, low_memory=False)
        print(f"✅ Successfully loaded {len(df)} rows from {DATA_FILE_PATH}.")
        
        # --- Data Cleaning & Final Preparation ---
        df['concession_creation_day'] = pd.to_datetime(df['concession_creation_day'], errors='coerce')
        df['ship_day'] = pd.to_datetime(df['ship_day'], errors='coerce')
        df.dropna(subset=['asin'], inplace=True)
        
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna('N/A')
            
        print("✅ Data pre-cleaned and ready for final loading.")

        print("\n--- Starting PostgreSQL Ingestion (Enriched Schema) ---")
        
        # --- 1. PRODUCTS TABLE (Now with Enriched Columns) ---
        product_columns = [
            'asin', 'ProductName', 'product_type', 'category', 'manufacturer_name', 
            'subcategory_code', 'subcategory_description', 'asp_band',
            'PurchaseIntent', 'MarketSegment', 'PredictedDiscountImpact'
        ]
        existing_product_cols = [col for col in product_columns if col in df.columns]
        products_df = df[existing_product_cols].drop_duplicates(subset=['asin']).copy()
        products_df.rename(columns={'ProductName': 'product_name'}, inplace=True)
        
        products_df.to_sql('products', con=engine, if_exists='append', index=False)
        print(f"✅ Loaded {len(products_df)} unique products with enriched details into 'products' table.")

        # --- 2. CONCESSIONS TABLE (Now with Enriched Columns) ---
        concession_columns = [
            'asin', 'customer_id', 'concession_creation_day', 'concession_reason',
            'Sentiment', 'ReturnTheme', 'SuggestedAction' 
        ]
        existing_concession_cols = [col for col in concession_columns if col in df.columns]
        concessions_df = df[df['total_units_conceded'] > 0][existing_concession_cols].copy()
        
        concessions_df.to_sql('concessions', con=engine, if_exists='append', index=False)
        print(f"✅ Loaded {len(concessions_df)} enriched records into 'concessions' table.")

        # --- 3. WEEKLY PERFORMANCE TABLE (Unchanged, but rebuilt for consistency) ---
        sales_df = df[(df['shipped_units'] > 0) & (df['ship_day'].notna())].copy()
        sales_df['week_start_date'] = pd.to_datetime(sales_df['ship_day']).dt.to_period('W-MON').dt.start_time

        weekly_performance = sales_df.groupby(['week_start_date', 'asin']).agg(
            total_units_sold=('shipped_units', 'sum')
        ).reset_index().fillna(0)
        weekly_performance['total_units_sold'] = weekly_performance['total_units_sold'].astype(int)
        
        weekly_performance.to_sql('weekly_performance', con=engine, if_exists='append', index=False)
        print(f"✅ Loaded {len(weekly_performance)} weekly sales summaries into 'weekly_performance' table.")
        
        print("--- ✅ Finished PostgreSQL Ingestion ---")

    except Exception as e:
        print(f"❌ ERROR during data processing or PostgreSQL ingestion: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n🎉 --- Full Data Ingestion Pipeline Finished Successfully --- 🎉")

if __name__ == "__main__":
    run()