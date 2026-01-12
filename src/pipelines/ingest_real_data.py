# src/pipelines/ingest_real_data.py
import os
import sys
import pandas as pd
import time
from sqlalchemy import text

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.database import engine, create_tables

DATA_FILE_PATH = os.path.join(project_root, 'data', 'enriched_data.csv')

def run():
    print(f"--- 🚀 Starting Ingestion Pipeline ---")
    create_tables() 
    
    try:
        df = pd.read_csv(DATA_FILE_PATH, low_memory=False)
        df.columns = df.columns.str.strip()
        
        # Standardize naming for the dataframe to match DB
        df.rename(columns={
            'ProductName': 'product_name',
            'Sentiment': 'sentiment',
            'ReturnTheme': 'return_theme',
            'SuggestedAction': 'suggested_action',
            'MarketSegment': 'market_segment',
            'PurchaseIntent': 'purchase_intent'
        }, inplace=True, errors='ignore')

        # Clean Dates
        df['concession_creation_day'] = pd.to_datetime(df['concession_creation_day'], errors='coerce')
        df['ship_day'] = pd.to_datetime(df['ship_day'], errors='coerce')
        df.dropna(subset=['asin'], inplace=True)
        
        # 1. Products
        print("DEBUG: Ingesting products...")
        product_cols = ['asin', 'product_name', 'manufacturer_name', 'product_type', 'asp_band']
        # Add fallback for missing columns
        for c in ['market_segment', 'purchase_intent', 'category']:
            if c in df.columns: product_cols.append(c)
        
        products_df = df[product_cols].drop_duplicates(subset=['asin']).copy()
        products_df.to_sql('products', con=engine, if_exists='append', index=False, method='multi')

        # 2. Concessions
        print("DEBUG: Ingesting concessions...")
        con_cols = ['asin', 'customer_id', 'concession_creation_day', 'concession_reason', 'sentiment', 'return_theme', 'suggested_action']
        concessions_df = df[df['total_units_conceded'] > 0][con_cols].copy()
        concessions_df.to_sql('concessions', con=engine, if_exists='append', index=False, method='multi')

        # 3. Performance
        print("DEBUG: Ingesting sales...")
        df['week_start_date'] = df['ship_day'].dt.to_period('W-MON').dt.start_time
        sales_agg = df.groupby(['week_start_date', 'asin'])['shipped_units'].sum().reset_index()
        sales_agg.rename(columns={'shipped_units': 'total_units_sold'}, inplace=True)
        sales_agg.to_sql('weekly_performance', con=engine, if_exists='append', index=False, method='multi')

        print(f"🎉 Ingestion Complete!")

    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    run()