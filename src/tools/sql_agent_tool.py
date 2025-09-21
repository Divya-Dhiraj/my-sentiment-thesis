# src/tools/sql_agent_tool.py
import os
import re
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate # Import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# We no longer need 'create_sql_query_chain'
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain.agents import tool
from ..database import engine

@tool
def sql_query_tool(query: str) -> str:
    """
    Use this tool to answer questions about the database by running SQL queries.
    It can handle natural language questions or direct SQL.
    """
    print("--- 🛠️ SELF-CORRECTING SQL QUERY TOOL ACTIVATED ---")
    
    db = SQLDatabase(engine)
    
    llm = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL_NAME"), 
        temperature=0.0, 
        stop=None
    )
    
    execute_query_tool = QuerySQLDatabaseTool(db=db)

    # --- THIS IS THE NEW, DEFINITIVE FIX ---
    # Instead of using the problematic 'create_sql_query_chain', we build our own
    # chain manually. This gives us full control and prevents the 'stop'
    # parameter from being added automatically.

    sql_prompt_template = """You are a PostgreSQL expert. Given an input question, create a syntactically correct PostgreSQL query to run.
    Only use the following tables:
    {table_info}

    Do not add any explanation, commentary, or any text other than the SQL query itself.

    Question: {question}
    SQL Query:"""

    # Create a PromptTemplate instance with the database schema partially filled in.
    sql_prompt = PromptTemplate.from_template(
        template=sql_prompt_template,
        partial_variables={"table_info": db.get_table_info()}
    )

    # Our new, custom chain for writing SQL queries.
    write_query_chain = sql_prompt | llm | StrOutputParser()
    # --- END OF THE NEW FIX ---


    def run_with_correction(sql_query: str) -> str:
        try:
            print(f"--- [SQL Tool] Attempting to execute SQL query... ---\n{sql_query}")
            return execute_query_tool.invoke({"query": sql_query})
        except Exception as e:
            error_message = str(e)
            print(f"--- [SQL Tool] ⚠️ Initial SQL query failed. Error: {error_message} ---")
            print("--- [SQL Tool] 🧠 Engaging SQL Corrector Agent... ---")
            
            corrector_prompt_template = """
            You are an expert SQL developer. You are given a database schema, a failed SQL query, and the resulting error message.
            Your task is to analyze the error and provide a corrected version of the SQL query.
            Respond with ONLY the corrected SQL query.
            Database Schema:{schema}
            Failed SQL Query:{query}
            Database Error Message:{error}
            Corrected SQL Query:
            """
            corrector_prompt = ChatPromptTemplate.from_template(corrector_prompt_template)
            corrector_chain = corrector_prompt | llm | StrOutputParser()
            db_schema = db.get_table_info()
            corrected_sql = corrector_chain.invoke({"schema": db_schema, "query": sql_query, "error": error_message})
            corrected_sql = corrected_sql.strip().replace("```sql", "").replace("```", "")
            print(f"--- [SQL Tool] ✅ Corrector Agent proposed a new query: ---\n{corrected_sql}")
            print("--- [SQL Tool] Retrying with the corrected SQL query... ---")
            return execute_query_tool.invoke({"query": corrected_sql})

    if re.match(r"\s*SELECT", query, re.IGNORECASE):
        print(f"--- [SQL Tool] Input is a direct SQL query. ---")
        return run_with_correction(query)
    else:
        print(f"--- [SQL Tool] Input is a natural language question. Generating SQL... ---")
        # Use our new, reliable chain
        generated_sql = write_query_chain.invoke({"question": query})
        generated_sql = generated_sql.strip().replace("```sql", "").replace("```", "")
        return run_with_correction(generated_sql)