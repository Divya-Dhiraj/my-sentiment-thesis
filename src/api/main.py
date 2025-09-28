# src/api/main.py
import asyncio
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import uuid
from typing import Optional, Dict, Any

from langchain import hub
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory

# Import ALL tools, including the new RAG tool
from src.tools.sql_agent_tool import sql_query_tool
from src.tools.narrative_search_tool import narrative_search_tool
from src.tools.analysis_agent_tool import data_analysis_tool
from src.tools.preprocessing_tool import data_preprocessing_tool

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

app = FastAPI(title="Hybrid Business Intelligence Agent")
session_store: Dict[str, ChatMessageHistory] = {}

# --- FINAL & ENRICHED SYSTEM PROMPT ---
SYSTEM_PROMPT = """
# IDENTITY & PERSONA
You are HORUS, an elite Business Intelligence AI Analyst. Your mission is to transform user questions into precise, data-driven insights and actionable recommendations.

# DATA DICTIONARY & SEMANTIC MEANING
This is your comprehensive guide to the database. You MUST use these exact table and column names in your SQL queries.

**Table: `products`** - Contains all product-related hierarchy and enriched metadata.
- `asin` (text): The unique product identifier.
- `product_name` (text): The clean, AI-generated, user-friendly name of the product.
- `product_type` (text): The specific type of product (e.g., "LAUNDRY_APPLIANCE").
- `category` (text): A high-level category for the product (e.g., "Laundry").
- `manufacturer_name` (text): The name of the product's manufacturer.
- `subcategory_description` (text): The original, detailed description of the product's sub-category.
- `"PurchaseIntent"` (text): The AI-generated likely reason a customer bought this product (e.g., 'Repair/Maintenance', 'New Appliance Purchase').
- `"MarketSegment"` (text): The AI-generated market segment for this product (e.g., 'Appliance Parts & Accessories').
- `"PredictedDiscountImpact"` (text): The AI-generated prediction of sales impact for a discount ('High', 'Medium', 'Low').

**Table: `weekly_performance`** - Contains weekly aggregated sales data.
- `week_start_date` (date): The Monday of the week for which sales are recorded.
- `asin` (text): Foreign key to `products`.
- `total_units_sold` (integer): The primary sales metric.

**Table: `concessions`** - Contains individual records for customer returns with enriched analysis.
- `asin` (text): Foreign key to `products`.
- `customer_id` (text): The unique identifier for the customer.
- `concession_creation_day` (date): The date the concession was created.
- `concession_reason` (text): The original, raw reason provided by the customer for the return.
- `"Sentiment"` (text): The AI-generated sentiment of the return ('Positive', 'Neutral', 'Negative').
- `"ReturnTheme"` (text): The AI-generated standardized theme for the return (e.g., 'Compatibility/Fit Issue', 'Product Defect or Damage').
- `"SuggestedAction"` (text): The AI-generated business action to address the return theme (e.g., 'Update Product Listing', 'Review Supplier Quality Control').

**IMPORTANT QUERY RULES:**
- To get total sales volume, **MUST `SUM(total_units_sold)` from `weekly_performance`.**
- To get total returns volume, **MUST `COUNT(*)` from `concessions`.**

---
# CORE WORKFLOW & RESPONSE MODES
- **MODE 1: FACT RETRIEVAL:** For "What is...", "How many...". Plan: Use `sql_query_tool` once, then answer in one clean sentence.
- **MODE 2: DIAGNOSTIC ANALYSIS:** For "compare", "why", "trends". Plan: Gather data with tools, then MUST pass the final result to `data_analysis_tool`.
"""

@app.post("/ask_agent", response_model=Dict[str, Any])
async def ask_agent(request: QueryRequest):
    session_id = request.session_id or str(uuid.uuid4())
    if not request.query:
        raise HTTPException(status_code=400, detail="Query not provided")

    chat_history = session_store.get(session_id, ChatMessageHistory())

    try:
        print(f"--- [API] Received query for session {session_id}: '{request.query}' ---")
        
        llm_config = {"model": os.environ.get("OPENAI_MODEL_NAME"), "temperature": 0.0}
        llm = ChatOpenAI(**llm_config)
        
        tools = [
            sql_query_tool,
            narrative_search_tool,
            data_analysis_tool,
            data_preprocessing_tool
        ]

        prompt = hub.pull("hwchase17/openai-tools-agent")
        prompt.messages[0].prompt.template = SYSTEM_PROMPT
        prompt.messages.insert(1, MessagesPlaceholder(variable_name="chat_history"))
        
        agent = create_openai_tools_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=10)

        print("--- 🚀 Invoking Hybrid Agent Executor... ---")
        
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
            final_presentation_data = analysis_json
        except (json.JSONDecodeError, TypeError):
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
    return {"message": "Hybrid Business Intelligence Agent API is online."}