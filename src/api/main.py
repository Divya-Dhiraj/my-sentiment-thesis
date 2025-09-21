# src/api/main.py
import asyncio
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import uuid
from typing import Optional, List, Dict, Any

from langchain import hub
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory

# Import the baseline set of tools
from src.tools.sql_agent_tool import sql_query_tool
from src.tools.semantic_search_tool import semantic_product_search
from src.tools.analysis_agent_tool import data_analysis_tool
from src.tools.preprocessing_tool import data_preprocessing_tool
from src.tools.re_ranking_tool import re_ranker_tool

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

app = FastAPI(title="Business Intelligence RAG Agent")
session_store: Dict[str, ChatMessageHistory] = {}

# --- FINAL & COMPLETE SYSTEM PROMPT ---
SYSTEM_PROMPT = """
# IDENTITY & PERSONA
You are HORUS, an elite Business Intelligence AI Analyst. Your mission is to transform user questions into precise, data-driven insights and actionable recommendations. You are authoritative, insightful, and always speak to an executive audience.

# DATA DICTIONARY & SEMANTIC MEANING
This is your comprehensive guide to the database. You MUST use these exact table and column names in your SQL queries.

**Table: `products`** - Contains all product-related hierarchy and metadata.
- `asin` (text): The unique product identifier.
- `product_name` (text): The generic display name of the product (e.g., "Major Appliances").
- `gl_product_group` (text): The top-level general ledger group for the product.
- `gl_name` (text): The business-specific product line name (e.g., "Major Appliances").
- `subcategory_code` (text): The specific code for the product's sub-category.
- `subcategory_description` (text): A detailed description of the product's sub-category (e.g., "Washing and Drying Machine accessories").
- `category` (text): A high-level category for the product (e.g., "Laundry").
- `manufacturer_name` (text): The name of the product's manufacturer.
- `product_type` (text): The specific type of product (e.g., "LAUNDRY_APPLIANCE").
- `asp_band` (text): The Average Selling Price band for the product (e.g., "Low ASP").

**Table: `weekly_performance`** - Contains weekly aggregated sales data.
- `asin` (text): Foreign key to `products`.
- `week_start_date` (date): The Monday of the week for which sales are recorded.
- `total_units_sold` (integer): The primary sales metric. Use this for all sales volume calculations.
- `shipped_cogs` (numeric): The total Cost of Goods Sold for units shipped in that week.
- `product_gms` (numeric): The total Gross Merchandise Sales for that week.

**Table: `concessions`** - Contains individual records for customer returns or concessions.
- `customer_id` (text): The unique identifier for the customer.
- `asin` (text): Foreign key to `products`.
- `fulfillment_channel` (text): How the product was shipped (e.g., "FBA" for Fulfilled by Amazon).
- `concession_creation_day` (date): The date the concession was created.
- `concession_reason` (text): The specific reason provided for the concession/return.
- `defect_category` (text): A higher-level category for the reason of the return (e.g., "Fit/Style Issue").
- `root_cause` (text): The underlying root cause of the return issue (e.g., "Fit/Style").
- `total_units_conceded` (integer): The number of units conceded in this event.
- `our_price` (numeric): The price of the item at the time of concession.
- `marketplace_id` (integer): The sales region. `4` means **Germany**.

**IMPORTANT QUERY RULES:**
- To get total sales volume, **MUST `SUM(total_units_sold)` from `weekly_performance`.**
- To get total returns volume, **MUST `SUM(total_units_conceded)` from `concessions`.**
- There is **NO `region` column**. Use `marketplace_id`.

---
# CORE WORKFLOW & RESPONSE MODES
- **MODE 1: FACT RETRIEVAL:** For "What is...", "How many...". Plan: Use `sql_query_tool` once, then answer in one clean sentence. DO NOT use `data_analysis_tool`.
- **MODE 2: DIAGNOSTIC ANALYSIS:** For "compare", "why", "trends". Plan: Gather data with tools, then MUST pass the final result to `data_analysis_tool`. This is your primary mode.

# DATA PRESENTATION MANDATE
When a user asks for a "table", "list", "breakdown", "chart", or "graph", you MUST use the `data_analysis_tool`. Ensure you structure the data into the `table_data` or `chart_data` fields in the tool's input. The frontend will render these directly.

- **QUERY REFINEMENT:** If a query is too broad, you MUST halt and propose an aggregated query to the user.
- **PREPROCESSING:** If a SQL result is large (>5 rows), you MUST use `data_preprocessing_tool` before the next step.
"""

@app.post("/ask_agent", response_model=Dict[str, Any])
async def ask_agent(request: QueryRequest):
    session_id = request.session_id or str(uuid.uuid4())
    if not request.query:
        raise HTTPException(status_code=400, detail="Query not provided")

    chat_history = session_store.get(session_id, ChatMessageHistory())

    try:
        print(f"--- [API] Received query for session {session_id}: '{request.query}' ---")
        
        llm_config = {"model": os.environ.get("OPENAI_MODEL_NAME"), "temperature": 0.0, "stop": None}
        llm = ChatOpenAI(**llm_config)
        
        tools = [
            semantic_product_search,
            sql_query_tool,
            data_analysis_tool,
            data_preprocessing_tool,
            re_ranker_tool
        ]

        prompt = hub.pull("hwchase17/openai-tools-agent")
        prompt.messages[0].prompt.template = SYSTEM_PROMPT
        prompt.messages.insert(1, MessagesPlaceholder(variable_name="chat_history"))
        
        agent = create_openai_tools_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

        print("--- 🚀 Invoking Tool Calling Agent Executor... ---")
        
        response = await asyncio.to_thread(
            agent_executor.invoke, 
            {"input": request.query, "chat_history": chat_history.messages}
        )
        
        chat_history.add_user_message(request.query)
        chat_history.add_ai_message(response.get("output", ""))
        session_store[session_id] = chat_history

        print("--- ✅ Agent execution finished. ---")
        
        final_output_str = response.get("output", "No analysis was generated.")
        
        final_presentation_data = {}
        try:
            analysis_json = json.loads(final_output_str)
            print("--- [Router] Complex analytical response detected. ---")
            final_presentation_data = analysis_json
        except (json.JSONDecodeError, TypeError):
            print("--- [Router] Simple fact-based response detected. ---")
            final_presentation_data = {"analysis_summary": final_output_str}
        
        step_details = []
        if "intermediate_steps" in response:
            for i, step in enumerate(response["intermediate_steps"]):
                action, observation = step
                tool_input_str = json.dumps(action.tool_input) if isinstance(action.tool_input, dict) else str(action.tool_input)
                step_details.append({"step": i + 1, "tool_name": action.tool, "tool_input": tool_input_str, "output": str(observation)})
        
        final_presentation_data["session_id"] = session_id
        final_presentation_data["steps"] = step_details
        return final_presentation_data

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")

@app.get("/")
def root():
    return {"message": "Business Intelligence RAG Agent API is online."}