# # src/tools/custom_tools.py
# import os
# import re
# import pandas as pd
# from functools import lru_cache

# from langchain.agents import tool
# from sqlalchemy import text
# from thefuzz import process

# from ..database import engine

# @lru_cache(maxsize=1)
# def get_all_product_info() -> pd.DataFrame:
#     """
#     Fetches all product info from the database, including name and type.
#     Uses lru_cache to run this query only once.
#     """
#     print("--- Caching product list from database for fuzzy matching... ---")
#     try:
#         return pd.read_sql_query("SELECT asin, product_name, product_type FROM products;", engine)
#     except Exception as e:
#         print(f"Error fetching product info: {e}")
#         return pd.DataFrame(columns=['asin', 'product_name', 'product_type'])

# def find_product_in_query(query: str) -> dict:
#     """
#     Identifies a product's ASIN and name from a query by searching against
#     both product_name and product_type for a more robust match.
#     """
#     products_df = get_all_product_info()
#     if products_df.empty:
#         return {"error": "No products found in the database."}

#     asin_match = re.search(r'\b(B[A-Z0-9]{9})\b', query, re.IGNORECASE)
#     if asin_match:
#         asin = asin_match.group(0).upper()
#         match = products_df[products_df['asin'] == asin]
#         if not match.empty:
#             return {"asin": asin, "name": match.iloc[0]['product_name']}
#         else:
#             return {"error": f"ASIN {asin} was mentioned, but it does not exist in the database."}

#     products_df['search_text'] = products_df['product_name'].fillna('') + " " + products_df['product_type'].fillna('')
#     product_lookup = {row['search_text']: {'asin': row['asin'], 'name': row['product_name']} for _, row in products_df.iterrows()}
    
#     best_match = process.extractOne(query, product_lookup.keys())
#     if best_match and best_match[1] >= 75:
#         matched_text = best_match[0]
#         product_info = product_lookup[matched_text]
#         print(f"Found fuzzy name match: '{product_info['name']}' (ASIN: {product_info['asin']}) with score {best_match[1]}")
#         return product_info

#     return {"error": "Could not identify a specific product from your query. Please be more specific or provide an ASIN."}


# @tool
# def query_business_database_tool(query: str) -> str:
#     """
#     Answers questions about business performance by querying the internal database.
#     Use this for any questions related to sales, revenue, returns, or concession reasons.
#     The tool automatically identifies the product from the query.
#     """
#     print(f"--- 🧠 Smart Tool received query: '{query}' ---")
#     product = find_product_in_query(query)
#     if "error" in product:
#         return product["error"]
    
#     q_lower = query.lower()
#     is_sales_query = any(k in q_lower for k in ['sale', 'sold', 'revenue', 'performance', 'price'])
#     is_concession_query = any(k in q_lower for k in ['return', 'concession', 'defect', 'complaint', 'reason'])

#     if not is_sales_query and not is_concession_query:
#         is_sales_query = True
#         is_concession_query = True

#     params = {'asin': product['asin']}
#     results = [f"📊 Analysis for Product: **{product['name']}** (ASIN: {product['asin']})\n"]

#     try:
#         with engine.connect() as conn:
#             if is_sales_query:
#                 sql = text("""
#                     SELECT 
#                         SUM(total_units_sold) as "Total Units Sold",
#                         TO_CHAR(AVG(average_selling_price), '999,999.00') as "Average Selling Price",
#                         SUM(total_units_conceded) as "Total Units Conceded"
#                     FROM weekly_performance
#                     WHERE asin = :asin
#                 """)
#                 sales_data = pd.read_sql(sql, conn, params=params)
#                 if not sales_data.empty and sales_data.iloc[0]['Total Units Sold'] is not None:
#                     results.append("### Performance Summary\n" + sales_data.to_markdown(index=False))
#                 else:
#                     results.append("No sales data found for this product.")

#             if is_concession_query:
#                 sql = text("""
#                     SELECT concession_reason as "Concession Reason", COUNT(*) as "Count"
#                     FROM concessions
#                     WHERE asin = :asin
#                     GROUP BY concession_reason
#                     ORDER BY "Count" DESC
#                     LIMIT 5
#                 """)
#                 concession_data = pd.read_sql(sql, conn, params=params)
#                 if not concession_data.empty:
#                     results.append("\n### Top 5 Concession Reasons\n" + concession_data.to_markdown(index=False))
#                 else:
#                     results.append("\nNo concession data found for this product.")
        
#         return "\n".join(results)

#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         return f"An error occurred while querying the database: {e}"











# src/tools/custom_tools.py
import pandas as pd
from typing import List
from langchain.agents import tool
from sqlalchemy import text
import re

from ..database import engine

@tool
def get_performance_data(asins: str) -> str:
    """
    Use this tool to get the performance summary for one or more products after you have their ASINs.
    The input MUST be a comma-separated string of ASINs (e.g., "B0123ABC,B0456DEF").
    """
    print(f"--- 📊 Performance Data Tool activated for ASINs: {asins} ---")
    try:
        # Parse the comma-separated string into a list
        asins_list = [asin.strip() for asin in asins.split(',')]
        params = {'asins': tuple(asins_list)}
        
        sql = text("""
            SELECT 
                p.product_name, p.asin, SUM(wp.total_units_sold) as "Total Units Sold",
                TO_CHAR(AVG(wp.average_selling_price), 'FM999,999.00') as "Average Selling Price",
                SUM(wp.total_units_conceded) as "Total Units Conceded"
            FROM weekly_performance wp
            JOIN products p ON wp.asin = p.asin
            WHERE wp.asin IN :asins
            GROUP BY p.product_name, p.asin
        """)
        with engine.connect() as conn:
            df = pd.read_sql(sql, conn, params=params)
        return f"Performance Data:\n{df.to_markdown(index=False)}" if not df.empty else "No performance data found for the given ASINs."
    except Exception as e:
        return f"Database error or invalid input format: {e}"

@tool
def get_concession_insights(asins: str) -> str:
    """
    Use this tool to find the top 5 return reasons for one or more products after you have their ASINs.
    The input MUST be a comma-separated string of ASINs (e.g., "B0123ABC,B0456DEF").
    """
    print(f"--- 💬 Concession Insights Tool activated for ASINs: {asins} ---")
    try:
        # Parse the comma-separated string into a list
        asins_list = [asin.strip() for asin in asins.split(',')]
        params = {'asins': tuple(asins_list)}
        
        sql = text("""
            SELECT 
                concession_reason as "Concession Reason", COUNT(*) as "Count"
            FROM concessions
            WHERE asin IN :asins
            GROUP BY concession_reason
            ORDER BY "Count" DESC
            LIMIT 5
        """)
        with engine.connect() as conn:
            df = pd.read_sql(sql, conn, params=params)
        return f"Top 5 Concession Reasons:\n{df.to_markdown(index=False)}" if not df.empty else "No concession data found for the given ASINs."
    except Exception as e:
        return f"Database error or invalid input format: {e}"