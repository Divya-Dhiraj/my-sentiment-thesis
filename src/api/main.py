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

# CORE LANGCHAIN 0.3 IMPORTS
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool

# TOOL IMPORTS
from src.tools.sql_agent_tool import sql_query_tool
from src.tools.narrative_search_tool import narrative_search_tool
from src.tools.analysis_agent_tool import data_analysis_tool
from src.tools.preprocessing_tool import data_preprocessing_tool

# --- ELITE VISUALIZATION MODULE ---
@tool
def data_visualization_tool(data: List[Dict[str, Any]], chart_type: str, title: str, x_label: str, y_label: str) -> str:
    """
    Elite visualization tool that creates high-fidelity charts using Seaborn.
    - data: A list of dictionaries (records) fetched from the database.
    - chart_type: One of ['bar', 'line', 'scatter', 'pie'].
    - x_label/y_label: The keys from the data to use for axes.
    Returns the image as a Base64 string prefixed with 'IMAGE_DATA:'.
    """
    try:
        # Clear any existing plots
        plt.clf()
        df = pd.DataFrame(data)
        
        # Set elite aesthetics
        sns.set_theme(style="whitegrid", palette="viridis")
        plt.figure(figsize=(10, 6))
        
        if chart_type.lower() == 'bar':
            sns.barplot(data=df, x=x_label, y=y_label)
        elif chart_type.lower() == 'line':
            sns.lineplot(data=df, x=x_label, y=y_label, marker='o', linewidth=2.5)
        elif chart_type.lower() == 'scatter':
            sns.scatterplot(data=df, x=x_label, y=y_label, s=100)
        elif chart_type.lower() == 'pie':
            plt.pie(df[y_label], labels=df[x_label], autopct='%1.1f%%', startangle=140)
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Buffer conversion
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return f"IMAGE_DATA:{img_str}"
    except Exception as e:
        return f"Visualization failed: {str(e)}"

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

app = FastAPI(title="Business Intelligence AI Agent")
session_store: Dict[str, ChatMessageHistory] = {}

SYSTEM_PROMPT = """
# IDENTITY & PERSONA
You are an elite Business Intelligence AI Analyst. Your mission is to provide data-driven insights and high-quality visualizations.

# DATA DICTIONARY
- Table `products`: Metadata for products (asin, product_name, category, etc.)
- Table `weekly_performance`: Sales data (asin, total_units_sold, week_start_date)
- Table `concessions`: Return data (asin, customer_id, ReturnTheme, Sentiment)

# WORKFLOW
1. DATA GATHERING: Use `sql_query_tool` to get the raw numbers.
2. ANALYSIS: Use `data_analysis_tool` for trends or diagnostic reasoning.
3. VISUALIZATION: If the user asks for a chart, graph, or plot, you MUST:
   - First get the data via SQL.
   - Then pass that data to `data_visualization_tool`.
"""

@app.post("/ask_agent", response_model=Dict[str, Any])
async def ask_agent(request: QueryRequest):
    session_id = request.session_id or str(uuid.uuid4())
    if not request.query:
        raise HTTPException(status_code=400, detail="Query not provided")

    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    chat_history = session_store[session_id]

    try:
        print(f"--- [API] Session {session_id} | Query: '{request.query}' ---")
        
        llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL_NAME", "gpt-4o"), temperature=0.0)
        
        # Tools including the new Elite Visualization module
        tools = [
            sql_query_tool,
            narrative_search_tool,
            data_analysis_tool,
            data_preprocessing_tool,
            data_visualization_tool
        ]

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
            return_intermediate_steps=True
        )

        response = await asyncio.to_thread(
            agent_executor.invoke, 
            {"input": request.query, "chat_history": chat_history.messages}
        )
        
        output_content = response.get("output", "")
        chat_history.add_user_message(request.query)
        chat_history.add_ai_message(output_content)

        # Build Elite Response Object
        final_presentation_data = {"session_id": session_id, "chart_base64": None}
        
        # Logic to extract the image from tool observations
        step_details = []
        if "intermediate_steps" in response:
            for action, observation in response["intermediate_steps"]:
                obs_str = str(observation)
                if obs_str.startswith("IMAGE_DATA:"):
                    final_presentation_data["chart_base64"] = obs_str.split("IMAGE_DATA:")[1]
                    display_output = "[Visual Chart Generated]"
                else:
                    display_output = obs_str

                step_details.append({
                    "tool": action.tool,
                    "input": str(action.tool_input),
                    "output": display_output
                })

        try:
            final_presentation_data.update(json.loads(output_content))
        except (json.JSONDecodeError, TypeError):
            final_presentation_data["analysis_summary"] = output_content
        
        final_presentation_data["steps"] = step_details
        return final_presentation_data

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

@app.get("/")
def root():
    return {"message": "Advanced BI Agent API is active."}