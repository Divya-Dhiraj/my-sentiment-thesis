# src/tools/sql_agent_tool.py

import os
import re
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain.agents import Tool

# Import the database engine from your existing database.py
from ..database import engine

def get_sql_query_tool() -> Tool:
    """
    This function creates an intelligent SQL Query Tool that can
    execute a query, and if it fails, it can self-correct and retry.
    """
    print("--- INITIALIZING SELF-CORRECTING SQL QUERY TOOL ---")
    
    db = SQLDatabase(engine)
    
    llm = ChatOpenAI(
        model=os.environ.get("NOVITA_MODEL_NAME"),
        api_key=os.environ.get("NOVITA_API_KEY"),
        base_url=os.environ.get("NOVITA_API_BASE_URL"),
        temperature=0.0
    )
    
    write_query_chain = create_sql_query_chain(llm, db)
    execute_query_tool = QuerySQLDataBaseTool(db=db)

    # --- FINAL SELF-CORRECTION LOGIC ---
    def run_with_correction(sql_query: str) -> str:
        """
        Executes a SQL query. If the result is an error message, it uses an LLM 
        to correct the query based on the error and retries once.
        """
        print(f"--- Attempting to execute SQL query... ---\n{sql_query}")
        # First, try to execute the query.
        result = execute_query_tool.invoke(sql_query)

        # Check if the result is a string and starts with "Error:".
        if isinstance(result, str) and result.strip().lower().startswith("error:"):
            print(f"--- ⚠️ Initial SQL query failed. Error: {result} ---")
            print("--- 🧠 Engaging SQL Corrector Agent... ---")

            corrector_prompt_template = """
            You are an expert SQL developer. You are given a database schema, a failed SQL query, and the resulting error message.
            Your task is to analyze the error and provide a corrected version of the SQL query.
            For example, if a query uses `asin = 'value1', 'value2'` which is invalid, you should correct it to use `asin IN ('value1', 'value2')`.
            Respond with ONLY the corrected SQL query. Do not include any other text, greetings, or explanations.

            Database Schema:
            {schema}

            Failed SQL Query:
            {query}

            Database Error Message:
            {error}

            Corrected SQL Query:
            """
            corrector_prompt = ChatPromptTemplate.from_template(corrector_prompt_template)
            corrector_chain = corrector_prompt | llm | StrOutputParser()

            db_schema = db.get_table_info()
            corrected_sql = corrector_chain.invoke({
                "schema": db_schema,
                "query": sql_query,
                "error": result 
            })

            corrected_sql = corrected_sql.strip().replace("```sql", "").replace("```", "")
            print(f"--- ✅ Corrector Agent proposed a new query: ---\n{corrected_sql}")
            
            print("--- Retrying with the corrected SQL query... ---")
            return execute_query_tool.invoke(corrected_sql)
        else:
            # If there was no error, return the result directly.
            return result

    def run_intelligent_sql_tool(query: str) -> str:
        """
        Inspects the input query. If it's a direct SQL query, it executes it.
        Otherwise, it generates SQL from the natural language question.
        Both paths use the self-correction mechanism.
        """
        if re.match(r"\s*SELECT", query, re.IGNORECASE):
            print(f"--- Input is a direct SQL query. ---")
            return run_with_correction(query)
        else:
            print(f"--- Input is a natural language question. Generating SQL... ---")
            generated_sql = write_query_chain.invoke({"question": query})
            # Clean up potential markdown formatting from the generator
            generated_sql = generated_sql.strip().replace("```sql", "").replace("```", "")
            return run_with_correction(generated_sql)

    sql_query_tool = Tool(
        name="SQL Query Tool",
        func=run_intelligent_sql_tool,
        description=(
            "Use this tool to answer questions about the database. "
            "It can handle natural language questions or direct SQL queries. "
            "It will automatically try to correct errors if a query fails."
        )
    )
    
    print("--- SELF-CORRECTING SQL QUERY TOOL INITIALIZED ---")
    return sql_query_tool

# Create a single instance of the tool to be imported
sql_query_tool = get_sql_query_tool()