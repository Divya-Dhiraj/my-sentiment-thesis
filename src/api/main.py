# src/api/main.py
import asyncio
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import uuid
from typing import Optional, List, Dict, Any

from langchain import hub
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory

# Import the decorated tool functions directly
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

def create_final_presentation(analysis_json: Dict[str, Any], llm_config: dict) -> Dict[str, Any]:
    print("--- ✨ PRESENTATION AGENT: Polishing final output... ---")
    if "No data was provided" in analysis_json.get("analysis_summary", "") or "No data found" in analysis_json.get("analysis_summary", ""):
        return analysis_json
        
    presenter_llm_config = llm_config.copy()
    presenter_llm_config["temperature"] = 0.1
    print(f"--- [Presentation Agent] Using LLM config: {presenter_llm_config} ---")
    
    presenter_prompt_template = """
    You are a senior business communications expert. Your job is to take a structured data analysis report (in JSON format)
    and rewrite the textual parts into a clean, concise, and professional summary for an executive audience.
    **Instructions:**
    1.  Read the `analysis_summary`. Rewrite it to be more narrative and less technical. Remove references to "fields", "records", or "data points". Focus on the business meaning.
    2.  Review the `key_insights` and weave them into your summary naturally.
    3.  Review the `actionable_recommendations`. Refine the language to be as clear and impactful as possible.
    4.  Return a JSON object with the polished `analysis_summary` and `actionable_recommendations`.
    **Data Analysis JSON:**
    {analysis_json}
    **Polished JSON Output:**
    """
    prompt = ChatPromptTemplate.from_template(presenter_prompt_template)
    parser = StrOutputParser()
    
    presenter_llm = ChatOpenAI(**presenter_llm_config)

    presentation_chain = prompt | presenter_llm | parser
    response_str = presentation_chain.invoke({"analysis_json": json.dumps(analysis_json, indent=2)})
    try:
        polished_text_data = json.loads(response_str)
        final_presentation = {
            "analysis_summary": polished_text_data.get("analysis_summary", analysis_json.get("analysis_summary", "")),
            "actionable_recommendations": polished_text_data.get("actionable_recommendations", []),
            "key_insights": analysis_json.get("key_insights", []),
            "data_quality_concerns": analysis_json.get("data_quality_concerns", []),
            "chart_data": analysis_json.get("chart_data", {})
        }
        print("--- ✅ PRESENTATION AGENT: Output polished successfully. ---")
        return final_presentation
    except json.JSONDecodeError:
        print("--- ⚠️ PRESENTATION AGENT: Failed to parse polished JSON, returning original analysis. ---")
        return analysis_json

# --- SYSTEM PROMPT WITH DATA BUG FIX INSTRUCTION ---
SYSTEM_PROMPT = """
You are a senior Business Intelligence analyst. Your primary goal is to provide accurate, data-driven answers and proactive insights to help users make informed business decisions.
You must speak like a professional analyst: be concise, confident, and focus on the business implications of the data.

**ANALYSIS & INSIGHTS RULES:**
1.  **Data Quality First:** Before answering the user's question, ALWAYS perform a quick sanity check of the data. If you notice anomalies (e.g., return rates over 100%, extremely low sample sizes, missing data), you MUST mention these as "Data Quality Concerns" in your final analysis.
2.  **Be Proactive:** Don't just answer the question. After providing a direct answer, you MUST suggest 2-3 specific, actionable follow-up questions or analyses that would provide deeper insight.
3.  **Use Your Tools:** You have access to a suite of tools. Use them efficiently and in the correct sequence to answer questions.

**ADVANCED RE-RANKING STRATEGY:**
For complex questions with both semantic (e.g., "popular", "poorly rated") and structured (e.g., "in the 'Laundry Appliance' category", "with sales > 1000") criteria, you MUST use the following three-step process:
1.  **Step 1: Semantic Search.** Use the `semantic_product_search` tool with the semantic part of the query to get a list of conceptually relevant products.
2.  **Step 2: SQL Search.** Use the `sql_query_tool` with the structured part of the query to get a list of products that meet the exact criteria.
3.  **Step 3: Re-rank.** Use the `re_ranker_tool` with the user's original query and the results from BOTH Step 1 and Step 2 to get a final, fused, and intelligently ranked list of products.
This strategy is crucial for accuracy on complex queries.

**DATABASE SCHEMA HINTS:**
- The main product identifier is the `asin` column.
- Sales data is in `weekly_performance` (`total_units_sold`).
- Return data is in `concessions`. To count total returns for a product, you must `COUNT(*)` from the `concessions` table. **DO NOT SUM `total_units_conceded` from the `weekly_performance` table for return analysis, as this is incorrect.**
- Return reasons are in `concessions` (`concession_reason`).
- Product info is in `products` (`product_name`, `clean_category`).
- Price is `average_selling_price` in `weekly_performance`.
- ALWAYS use the `sql_query_tool` for specific data, numbers, or aggregations.

**QUERY STRATEGY HINTS:**
- Construct single, efficient SQL queries that join tables to get the final answer in one step.
- To get both sales and return counts, you may need to use subqueries or Common Table Expressions (CTEs) that aggregate from `weekly_performance` and `concessions` separately before joining them on the `asin`.
"""

@app.post("/ask_agent", response_model=Dict[str, Any])
async def ask_agent(request: QueryRequest):
    session_id = request.session_id or str(uuid.uuid4())
    if not request.query:
        raise HTTPException(status_code=400, detail="Query not provided")

    chat_history = session_store.get(session_id, ChatMessageHistory())

    try:
        print(f"--- [API] Received query for session {session_id}: '{request.query}' ---")
        
        llm_config = {
            "model": os.environ.get("OPENAI_MODEL_NAME"),
            "temperature": 0.0,
            "stop": None
        }
        
        print(f"--- [API] Initializing main agent LLM with config: {llm_config} ---")
        llm = ChatOpenAI(**llm_config)
        print("--- [API] LLM Initialized Successfully ---")
        
        tools = [
            semantic_product_search,
            sql_query_tool,
            data_analysis_tool,
            data_preprocessing_tool,
            re_ranker_tool 
        ]

        prompt = hub.pull("hwchase17/openai-tools-agent")
        original_system_template = prompt.messages[0].prompt.template
        prompt.messages[0].prompt.template = SYSTEM_PROMPT + "\n\n" + original_system_template
        prompt.messages.insert(1, MessagesPlaceholder(variable_name="chat_history"))
        
        print("--- [API] Creating OpenAI Tools Agent ---")
        agent = create_openai_tools_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
        print("--- [API] Agent Executor created successfully ---")

        print("--- 🚀 Invoking Tool Calling Agent Executor... ---")
        
        response = await asyncio.to_thread(
            agent_executor.invoke, 
            {"input": request.query, "chat_history": chat_history.messages}
        )
        
        chat_history.add_user_message(request.query)
        chat_history.add_ai_message(response.get("output", ""))
        session_store[session_id] = chat_history

        print("--- ✅ Agent execution finished. ---")
        
        final_analysis_str = response.get("output", "No analysis was generated.")
        try:
            if "```json" in final_analysis_str:
                json_part = final_analysis_str.split("```json")[1].split("```")[0]
                analysis_json = json.loads(json_part)
            else:
                analysis_json = json.loads(final_analysis_str)
        except (json.JSONDecodeError, TypeError, IndexError):
            analysis_json = {"analysis_summary": final_analysis_str}
        
        final_presentation_data = create_final_presentation(analysis_json, llm_config)
        
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