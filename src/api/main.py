# src/api/main.py
import asyncio
import json
import os
import uuid
import io
import base64
from typing import Optional, Dict, Any, List

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool

from src.tools.sql_agent_tool import fetch_smartphone_data
from src.tools.analysis_agent_tool import data_analysis_tool
from src.tools.preprocessing_tool import data_preprocessing_tool

@tool
def data_visualization_tool(data: List[Dict[str, Any]], chart_type: str, title: str, x_label: str, y_label: str) -> str:
    """Creates Seaborn charts. Input data must be a list of dictionaries."""
    try:
        plt.clf()
        df = pd.DataFrame(data)
        sns.set_theme(style="whitegrid", palette="viridis")
        plt.figure(figsize=(10, 6))
        if chart_type.lower() == 'bar': sns.barplot(data=df, x=x_label, y=y_label)
        elif chart_type.lower() == 'line': sns.lineplot(data=df, x=x_label, y=y_label, marker='o')
        elif chart_type.lower() == 'pie': plt.pie(df[y_label], labels=df[x_label], autopct='%1.1f%%')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return f"IMAGE_DATA:{img_str}"
    except Exception as e: return f"Visualization failed: {str(e)}"

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

app = FastAPI(title="HORUS BI System")
session_store: Dict[str, ChatMessageHistory] = {}

SYSTEM_PROMPT = """
# IDENTITY
You are HORUS, a Senior BI Analyst. You analyze a million-row smartphone dataset.

# STRICT TOOL RULE
- DO NOT attempt to write SQL code.
- To get data, simply pass the user's question in plain English to the `fetch_smartphone_data` tool.
- Example: If the user asks for sales, call `fetch_smartphone_data(question="Total sales for Samsung in 2025")`.

# WORKFLOW
1. Call `fetch_smartphone_data` with the English question.
2. Use `data_analysis_tool` to explain the results.
3. Use `data_visualization_tool` for charts.

DEBUG: HORUS simplified workflow active. Direct SQL generation by the agent is DISABLED.
"""

@app.post("/ask_agent", response_model=Dict[str, Any])
async def ask_agent(request: QueryRequest):
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in session_store: session_store[session_id] = ChatMessageHistory()
    chat_history = session_store[session_id]

    try:
        llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL_NAME", "gpt-4o"), temperature=0.0)
        tools = [fetch_smartphone_data, data_analysis_tool, data_preprocessing_tool, data_visualization_tool]

        prompt = hub.pull("hwchase17/openai-tools-agent")
        prompt.messages[0].prompt.template = SYSTEM_PROMPT
        if "chat_history" not in [getattr(m, "variable_name", None) for m in prompt.messages]:
            prompt.messages.insert(1, MessagesPlaceholder(variable_name="chat_history"))

        agent = create_openai_tools_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, 
            handle_parsing_errors=True,
            max_iterations=5 # Stop the loop early if it fails
        )

        response = await asyncio.to_thread(agent_executor.invoke, {"input": request.query, "chat_history": chat_history.messages})
        
        output_content = response.get("output", "")
        chat_history.add_user_message(request.query)
        chat_history.add_ai_message(output_content)

        final_data = {"session_id": session_id, "chart_base64": None, "analysis_summary": output_content}
        return final_data
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root(): return {"message": "Agent Ready"}