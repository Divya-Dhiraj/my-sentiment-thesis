# src/api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import pandas as pd
from typing import Optional
import uuid
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

# Import all of your powerful, custom tools
from src.tools.custom_tools import financial_data_tool, review_rag_tool, structured_sentiment_analyzer, browse_website_tool, plot_sales_trend, web_search_tool, extract_web_data_tool
from src.database import get_db_connection
# Set up the cache globally
set_llm_cache(InMemoryCache())
# --- Pydantic model for the request body ---
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

app = FastAPI(title="Dynamic & Stateful Multi-Agent System")

# This will act as our simple, in-memory session store.
SESSION_MEMORY = {}

def get_agent_for_session(session_id: str):
    """
    Creates or retrieves an agent executor for a given session ID.
    This ensures each conversation has its own separate memory and context.
    """
    
    # --- 1. Manage Conversational Memory ---
    if session_id not in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    memory = SESSION_MEMORY[session_id]

    # --- 2. Fetch Dynamic Context from the Database ---
    try:
        conn = get_db_connection()
        products_df = pd.read_sql_query("SELECT product_id, product_name FROM products;", conn)
        conn.close()
        product_context = "\n".join([f"- {row['product_name']} (ID: {row['product_id']})" for _, row in products_df.iterrows()])
    except Exception as e:
        print(f"Warning: Could not fetch dynamic product context. Error: {e}")
        product_context = "No specific products found in database."

    # --- 3. Define Tools and LLM ---
    #llm = ChatOllama(model="my-phi3", base_url=os.environ.get("OLLAMA_BASE_URL"))
        # +++ ADD THIS +++
    llm = ChatOpenAI(
        model="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
        api_key=os.environ.get("NOVITA_API_KEY"),
        base_url=os.environ.get("NOVITA_BASE_URL"),
        temperature=0.7 # Optional: Adjust as needed
    )
    
    tools = [financial_data_tool, review_rag_tool, structured_sentiment_analyzer, browse_website_tool, plot_sales_trend, web_search_tool, extract_web_data_tool]

    # --- 4. Create the Final, Dynamically Populated Prompt ---
    prompt = PromptTemplate.from_template("""
    You are a master business analyst agent. Your primary function is to use the tools provided to you to find data and answer the user's question. You must act as a rational, data-driven expert.

    LIVE CONTEXT FROM DATABASE:
    ---------------------------
    You are currently tracking the following products:
    {product_context}

    CONVERSATION HISTORY:
    ---------------------
    {chat_history}

    TOOLS:
    ------
    You have access to the following tools:
    {tools}

    RULES:
    - You MUST use your tools to find information. Do not rely on your pre-trained knowledge.
    - The input for a tool must be as simple and direct as possible. For a product ID, use only the ID like "XPS15-9540".
    - After using a tool, analyze the 'Observation' and decide if you need another tool or if you can now answer the question.
    - If you need to find information on the internet that is not in your internal databases (like current prices on specific retailer sites, news, or external specs), first use 'web_search_tool' to find relevant URLs.
    - If you have a specific URL and need to extract a specific piece of information from it, use 'extract_web_data_tool' with a CSS selector. You will need to infer common CSS selectors for prices or data based on typical webpage structures (e.g., '.price', '#product-price', 'span[itemprop="price"]').

    Use the following format for your thought process:

    Question: The user's original input question
    Thought: Your reasoning on what to do next to answer the question.
    Action: The name of the tool to use, which must be one of [{tool_names}]
    Action Input: The simple input for the chosen tool.
    Observation: The result returned from the tool.
    ... (this Thought/Action/Action Input/Observation can repeat multiple times)
    Thought: I have now gathered all the necessary information from my tools.
    Final Answer: Your final, synthesized, data-driven answer to the user's original question.

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}
    """)


    # --- 5. Create and Return the Agent Executor ---
    # We partially format the prompt with the context we just fetched
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

# --- Main API Endpoint ---
@app.post("/ask_agent")
async def ask_master_agent(request: QueryRequest):
    """Receives a query, gets the correct agent for the session, and gets an answer."""
    session_id = request.session_id or str(uuid.uuid4())
    print(f"--- Handling request for Session ID: {session_id} ---")

    if not request.query:
        raise HTTPException(status_code=400, detail="Query not provided")
    
    try:
        agent_executor = get_agent_for_session(session_id)
        
        response = await agent_executor.ainvoke({
            "input": request.query,
            "chat_history": SESSION_MEMORY[session_id].chat_memory.messages
        })
        
        return {"response": response.get("output"), "session_id": session_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")

@app.get("/")
def root():
    return {"message": "Dynamic & Stateful Multi-Agent System API is online."}