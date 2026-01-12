# src/tools/sql_agent_tool.py
import os
import time
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain.agents import tool
from ..database import engine

@tool
def fetch_smartphone_data(question: str) -> str:
    """
    Answers business questions about smartphone sales, trends, and return reasons.
    Input: A natural language question in plain English (e.g., 'Total Samsung sales in 2025').
    DO NOT provide SQL code as input.
    """
    print(f"\n--- 🛠️ DEBUG: Fetching Data for Question: '{question}' ---")
    
    db = SQLDatabase(engine)
    llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL_NAME", "gpt-4o"), temperature=0.0)
    execute_query_tool = QuerySQLDatabaseTool(db=db)

    # THE GOLDEN SCHEMA: This is the ONLY source of truth for table/column names.
    sql_prompt_template = """You are a PostgreSQL expert. Use ONLY these tables and columns:

    1. Table 'products':
       - asin (TEXT, Primary Key)
       - product_name (TEXT)
       - manufacturer_name (TEXT) -> Use for brand filters (e.g., 'Apple', 'Samsung')
       - market_segment (TEXT) -> (e.g., 'Flagship', 'Mid-Tier')

    2. Table 'weekly_performance':
       - week_start_date (DATE) -> The ONLY date column for sales trends
       - asin (TEXT) -> Foreign Key to 'products'
       - total_units_sold (INTEGER) -> Use this for all sales volume/counts

    3. Table 'concessions':
       - asin (TEXT) -> Foreign Key to 'products'
       - concession_creation_day (DATE)
       - concession_reason (TEXT)
       - sentiment (TEXT)
       - return_theme (TEXT)
       - suggested_action (TEXT)
       - search_vector (TSVECTOR) -> Use for text/narrative searches

    CRITICAL RULES:
    - To filter by Brand, JOIN 'products' with the target table on 'asin'.
    - Use 'total_units_sold' for all sales volume.
    - Use 'week_start_date' for all time/date filters.
    - ALWAYS use 'LIMIT 50'.
    - Return ONLY the raw SQL query. No explanation.

    Question: {question}
    SQL Query:"""

    sql_prompt = PromptTemplate.from_template(sql_prompt_template)
    write_query_chain = sql_prompt | llm | StrOutputParser()

    try:
        # Step 1: Generate SQL inside the tool (Hidden from the Agent)
        generated_sql = write_query_chain.invoke({"question": question})
        sql_clean = generated_sql.strip().replace("```sql", "").replace("```", "")
        
        print(f"DEBUG: [SQL Engine] Generated Query:\n{sql_clean}")
        
        # Step 2: Execute
        start_time = time.time()
        result = execute_query_tool.invoke({"query": sql_clean})
        
        print(f"DEBUG: [SQL Engine] Success in {time.time() - start_time:.4f}s")
        
        if not result or result == "[]":
            return "No data found. Please try a broader date range or check brand spelling."
        
        return result
    except Exception as e:
        print(f"⚠️ DEBUG: [SQL Engine] Failure: {str(e)}")
        return f"Database error: {str(e)}"