# src/api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import pandas as pd
from typing import Optional
import uuid
from langchain_community.cache import InMemoryCache
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from src.tools.custom_tools import financial_data_tool, review_rag_tool, structured_sentiment_analyzer, browse_website_tool, plot_sales_trend, web_search_tool, extract_web_data_tool
from src.database import engine

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

app = FastAPI(title="Dynamic & Stateful Multi-Agent System")

SESSION_MEMORY = {}

def get_agent_for_session(session_id: str):
    """Creates or retrieves an agent executor for a given session ID."""
    if session_id not in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = ConversationBufferWindowMemory(
            k=10, memory_key="chat_history", return_messages=True
        )
    memory = SESSION_MEMORY[session_id]

    try:
        products_df = pd.read_sql_query("SELECT asin, product_name FROM products;", engine)
        product_context = "\n".join([f"- {row['product_name']} (ASIN: {row['asin']})" for _, row in products_df.iterrows()])
    except Exception as e:
        print(f"Warning: Could not fetch dynamic product context. Error: {e}")
        product_context = "No specific products found in database."

    llm = ChatOpenAI(
        model="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
        api_key=os.environ.get("NOVITA_API_KEY"),
        base_url=os.environ.get("NOVITA_API_BASE_URL"),
        temperature=0.7,
        cache=InMemoryCache()
    )
    
    tools = [financial_data_tool, review_rag_tool, structured_sentiment_analyzer, browse_website_tool, plot_sales_trend, web_search_tool, extract_web_data_tool]

    # --- THIS PROMPT HAS THE CORRECTED SECTION ORDER ---
    prompt = PromptTemplate.from_template("""
    You are a world-class Senior Business Intelligence Analyst. Your primary mission is to provide actionable insights by deeply analyzing our internal business data. Your greatest strength is correlating financial performance with customer sentiment to uncover the "why" behind the "what".

    **CORE WORKFLOW:**
    1.  **Deconstruct the Query:** First, understand the user's core question. Identify the specific product (using fuzzy matching) and the time frame (using natural language date parsing) they are asking about.
    2.  **Prioritize Internal Data:** Your first and main priority is to use the internal database tools. For any query about product performance, you should almost always start by using `financial_data_tool` and `review_rag_tool` in sequence.
    3.  **Synthesize and Correlate:** After gathering data, your main task is to connect the dots. In your final thought before the answer, explicitly state the relationship between the financial data and the customer review data.

    **TOOL USAGE STRATEGY:**

    * **Primary Tools (Use these first for most queries):** `financial_data_tool`, `review_rag_tool`
    * **Secondary/On-Demand Tools (Use ONLY when explicitly asked):**
        * `plot_sales_trend`: Use only for "plot", "graph", "chart", "visualize". Report that the file was saved.
        * `web_search_tool`, `browse_website_tool`, `extract_web_data_tool`: Use only for "competitor prices", "market comparison", or external news.
    * **General Knowledge Fallback:** If a tool can't find a product or if the query is general (e.g., "What is a KPI?"), answer from your base knowledge.

    **TOOLS:**
    ------
    You have access to the following tools:
    {tools}

    **RESPONSE FORMAT & THOUGHT PROCESS:**
    Use the following format. For data-driven answers, you MUST structure your final response in the specified markdown format below, including the source tags.

    Question: The user's original input question
    Thought: Your reasoning on what to do next to answer the question.
    Action: The name of the tool to use, which must be one of [{tool_names}]
    Action Input: The simple input for the chosen tool.
    Observation: The result returned from the tool.
    ... (this Thought/Action/Action Input/Observation can repeat)
    Thought: I have now gathered all the necessary information and will format it according to the response instructions.
    Final Answer: [Your final, structured answer formatted EXACTLY as instructed below]

    ---
    **## 📈 Financial KPIs**
    * **Sales Performance:** [Summarize sales performance and add `(Source: Internal Database)`.]
    * **Pricing & Discounting:** [State the average selling price and add `(Source: Internal Database)`.]
    * **Key Metric:** [Calculate a key metric, e.g., "Total Revenue: $Z `(Source: Internal Database)`."]

    **## 💬 Customer Sentiment**
    * **Overall Rating:** [State the average review rating and add `(Source: Internal Database)`.]
    * **Positive Themes:** [List 1-2 key features customers are praising and add `(Source: Internal Reviews)`.]
    * **Negative Themes:** [List 1-2 key issues customers are complaining about and add `(Source: Internal Reviews)`.]

    **## 🧠 Correlated Analysis & Actionable Suggestions**
    * **Key Insight:** [Synthesize the link between the financial data and customer sentiment. Example: "A 5% drop in average rating in May, driven by complaints about battery life, directly preceded a 15% decline in sales in June."]
    * **Business Recommendation:** [Provide a clear, actionable suggestion based on the analysis.]
    ---

    **BEGIN!**

    CONVERSATION HISTORY:
    ---------------------
    {chat_history}

    Question: {input}
    Thought:{agent_scratchpad}
    """)
    
    partial_prompt = prompt.partial(product_context=product_context)
    
    agent = create_react_agent(llm, tools, partial_prompt)
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=15
    )
    return agent_executor

@app.post("/ask_agent")
async def ask_master_agent(request: QueryRequest):
    """Receives a query, gets the correct agent for the session, and gets an answer."""
    session_id = request.session_id or str(uuid.uuid4())
    print(f"--- Handling request for Session ID: {session_id} ---")

    if not request.query:
        raise HTTPException(status_code=400, detail="Query not provided")
    
    try:
        agent_executor = get_agent_for_session(session_id)
        response = await agent_executor.ainvoke({"input": request.query})
        return {"response": response.get("output"), "session_id": session_id}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")

@app.get("/")
def root():
    return {"message": "Dynamic & Stateful Multi-Agent System API is online."}