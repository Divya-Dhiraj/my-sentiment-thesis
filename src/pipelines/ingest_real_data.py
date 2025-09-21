# src/pipelines/ingest_real_data.py
import os
import sys
import pandas as pd
from sqlalchemy import text
import chromadb
from sentence_transformers import SentenceTransformer
import json
import numpy as np

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.database import engine, create_tables
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

DATA_FILE_PATH = os.path.join(project_root, 'data', 'processed_data.csv')

def generate_category_map(product_types: list) -> dict:
    print("--- 🤖 Engaging AI Categorizer Agent... ---")
    
    target_categories = [
        'Water Filter', 'Dishwasher Accessory', 'Laundry Appliance', 
        'Cleaning Agent', 'Lighting', 'Cooking Accessory', 'General Accessory'
    ]
    prompt_template = """
    You are an expert data categorization AI. Your task is to map a list of raw product types to a predefined list of clean categories.
    You must return a single, valid JSON object that represents this mapping. Return a valid JSON object.
    **Predefined Clean Categories:**
    {categories}
    **Raw Product Types to Map:**
    {product_types}
    **Instructions:**
    - For each raw product type, choose the BEST fitting category from the predefined list.
    - The output MUST be a single JSON object where keys are the raw product types and values are the chosen clean categories.
    - Do not include any other text or explanations in your response.
    **JSON Output:**
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL_NAME"), temperature=0.0, model_kwargs={"response_format": {"type": "json_object"}})
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"categories": ", ".join(target_categories), "product_types": ", ".join(product_types)})
    try:
        category_map = json.loads(response)
        print("--- ✅ AI Categorizer finished. Mapping created successfully. ---")
        return category_map
    except json.JSONDecodeError:
        print(f"--- ❌ ERROR: AI Categorizer failed to produce valid JSON. ---")
        return {}

def create_product_summary(df: pd.DataFrame) -> pd.DataFrame:
    print("--- 📊 Pre-aggregating data to create product summaries... ---")
    agg_functions = {
        'gl_name': 'first', 'product_type': 'first', 'clean_category': 'first',
        'manufacturer_name': 'first', 'our_price': 'mean', 'shipped_units': 'sum',
        'total_units_conceded': 'sum',
        'concession_reason': lambda x: x.mode()[0] if not x.mode().empty else 'N/A',
        'defect_category': lambda x: x.mode()[0] if not x.mode().empty else 'N/A',
        'root_cause': lambda x: x.mode()[0] if not x.mode().empty else 'N/A'
    }
    product_summary = df.groupby('asin').agg(agg_functions).reset_index()
    product_summary.rename(columns={
        'shipped_units': 'total_units_sold', 'concession_reason': 'top_concession_reason',
        'defect_category': 'top_defect_category', 'root_cause': 'top_root_cause'
    }, inplace=True)
    
    # Ensure division by zero is handled gracefully
    product_summary['return_rate'] = np.where(
        product_summary['total_units_sold'] > 0,
        product_summary['total_units_conceded'] / product_summary['total_units_sold'],
        0
    )
    product_summary['return_rate'] = product_summary['return_rate'].fillna(0)
    
    print(f"--- ✅ Created summary for {len(product_summary)} unique products. ---")
    return product_summary

def run(limit: int = None):
    if limit:
        print(f"--- 🚀 Starting FAST Data Ingestion Pipeline (LIMITED to {limit} rows) ---")
    else:
        print("--- 🚀 Starting Full Data Ingestion Pipeline (SQL + Vector) ---")
    create_tables()
    try:
        df = pd.read_csv(DATA_FILE_PATH, sep='\t', low_memory=False)
        if limit:
            print(f"--- ⚠️ Limiting DataFrame to first {limit} rows for quick testing. ---")
            df = df.head(limit)
        str_cols = ['gl_name', 'product_type', 'manufacturer_name', 'concession_reason', 'defect_category', 'root_cause']
        for col in str_cols:
            df[col] = df[col].fillna('N/A')
        df['ship_day'] = pd.to_datetime(df['ship_day'], errors='coerce')
        df.dropna(subset=['ship_day'], inplace=True)
        print("✅ Data loaded and dates parsed successfully.")
    except FileNotFoundError:
        print(f"❌ ERROR: Data file not found at {DATA_FILE_PATH}.")
        return
        
    unique_product_types = df['product_type'].unique().tolist()
    category_mapping = generate_category_map(unique_product_types)
    if not category_mapping:
        print("❌ Halting ingestion due to categorization failure.")
        return
    df['clean_category'] = df['product_type'].map(category_mapping).fillna('Other Accessory')
    print("✅ Added 'clean_category' to the dataset using the AI-generated map.")

    print("\n--- Starting PostgreSQL Ingestion ---")
    try:
        products_df = df[['asin', 'gl_name', 'product_type', 'clean_category']].drop_duplicates(subset=['asin']).copy()
        products_df.rename(columns={'gl_name': 'product_name'}, inplace=True)
        products_df.to_sql('products', con=engine, if_exists='append', index=False)
        print(f"✅ Loaded {len(products_df)} unique products into 'products' table.")
        
        # --- DATA BUG FIX: Correctly process concessions and orders ---
        # A concession record is defined as any row where 'total_units_conceded' is greater than 0.
        print("--- [Data Fix] Separating orders from concessions based on 'total_units_conceded' column. ---")
        concessions_df = df[df['total_units_conceded'] > 0].copy()
        orders_df = df[df['total_units_conceded'] == 0].copy()
        print(f"--- [Data Fix] Found {len(orders_df)} order records and {len(concessions_df)} concession records. ---")

        concessions_columns = ['asin', 'ship_day', 'concession_reason', 'defect_category', 'root_cause']
        concessions_df[concessions_columns].to_sql('concessions', con=engine, if_exists='append', index=False)
        print(f"✅ Loaded {len(concessions_df)} records into 'concessions' table.")
        
        # Now, create weekly summaries from the correctly separated dataframes
        weekly_sales = orders_df.groupby([pd.Grouper(key='ship_day', freq='W-MON'), 'asin']).agg(
            total_units_sold=('shipped_units', 'sum'),
            average_selling_price=('our_price', 'mean')
        ).reset_index()
        
        weekly_returns = concessions_df.groupby([pd.Grouper(key='ship_day', freq='W-MON'), 'asin']).agg(
            total_units_conceded=('total_units_conceded', 'sum')
        ).reset_index()
        
        weekly_performance_df = pd.merge(weekly_sales, weekly_returns, on=['ship_day', 'asin'], how='outer').fillna(0)
        
        # Ensure integer types for unit counts
        weekly_performance_df['total_units_sold'] = weekly_performance_df['total_units_sold'].astype(int)
        weekly_performance_df['total_units_conceded'] = weekly_performance_df['total_units_conceded'].astype(int)

        weekly_performance_df.rename(columns={'ship_day': 'week_start_date'}, inplace=True)
        performance_columns = ['week_start_date', 'asin', 'total_units_sold', 'average_selling_price', 'total_units_conceded']
        weekly_performance_df[performance_columns].to_sql('weekly_performance', con=engine, if_exists='append', index=False)
        print(f"✅ Loaded {len(weekly_performance_df)} weekly summary records into 'weekly_performance' table.")
        print("--- ✅ Finished PostgreSQL Ingestion ---")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ ERROR during PostgreSQL ingestion: {e}")
        return

    print("\n--- Starting ChromaDB Vector Ingestion (with Enriched Data) ---")
    try:
        embedding_model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')
        product_summary_df = create_product_summary(df)
        documents, metadatas, asins = [], [], []
        print("--- 📝 Generating enriched narratives and metadata for each product... ---")
        for _, row in product_summary_df.iterrows():
            narrative = (
                f"Product ASIN {row['asin']} is named '{row['gl_name']}'. "
                f"It is a {row['product_type']} from manufacturer '{row['manufacturer_name']}', and falls into the '{row['clean_category']}' category. "
                f"This product has an average selling price of {row['our_price']:.2f} EUR. "
                f"It has sold a total of {int(row['total_units_sold'])} units and had {int(row['total_units_conceded'])} units returned, giving it a return rate of {row['return_rate']:.2%}. "
                f"The most common reason for return is '{row['top_concession_reason']}', with a typical defect category of '{row['top_defect_category']}' and root cause of '{row['top_root_cause']}'."
            )
            documents.append(narrative)
            avg_price = row['our_price']
            total_sold = row['total_units_sold']
            total_conceded = row['total_units_conceded']
            return_rate = row['return_rate']
            metadata = {
                "clean_category": str(row['clean_category']), "manufacturer_name": str(row['manufacturer_name']),
                "product_type": str(row['product_type']), "average_price": float(avg_price) if not np.isnan(avg_price) else 0.0,
                "total_units_sold": int(total_sold) if not np.isnan(total_sold) else 0,
                "total_units_conceded": int(total_conceded) if not np.isnan(total_conceded) else 0,
                "return_rate": float(return_rate) if not np.isnan(return_rate) else 0.0,
                "top_concession_reason": str(row['top_concession_reason'])
            }
            metadatas.append(metadata)
            asins.append(row['asin'])
        
        print(f"--- ✅ Generated {len(documents)} enriched documents. Now creating embeddings... ---")
        embeddings = embedding_model.encode(documents, show_progress_bar=True).tolist()
        chroma_client = chromadb.HttpClient(host="chroma", port=8000)
        collection_name = "products"
        try:
            chroma_client.delete_collection(name=collection_name)
            print(f"Deleted old ChromaDB collection: '{collection_name}'")
        except Exception:
            print(f"ChromaDB collection '{collection_name}' did not exist, creating new.")
        collection = chroma_client.create_collection(name=collection_name)
        
        batch_size = 1000 
        for i in range(0, len(asins), batch_size):
            batch_ids, batch_embeddings, batch_metadatas, batch_documents = asins[i:i+batch_size], embeddings[i:i+batch_size], metadatas[i:i+batch_size], documents[i:i+batch_size]
            print(f" 	- Adding batch {i//batch_size + 1} to ChromaDB...")
            collection.add(ids=batch_ids, embeddings=batch_embeddings, metadatas=batch_metadatas, documents=batch_documents)
        print("--- ✅ Finished ChromaDB Vector Ingestion ---")

    except Exception as e:
        print(f"❌ ERROR during ChromaDB ingestion: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n🎉 --- Full Data Ingestion Pipeline Finished Successfully --- 🎉")

if __name__ == "__main__":
    run()
