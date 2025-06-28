# src/api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate

# Import the tools you created
from src.tools.custom_tools import financial_data_tool, review_rag_tool, structured_sentiment_analyzer, web_scraper_tool

class QueryRequest(BaseModel):
    query: str

app = FastAPI(title="Multi-Agent AI System")

# --- Initialize the LLM ---
llm = ChatOllama(
    model="my-phi3",
    base_url=os.environ.get("OLLAMA_BASE_URL")
)

# --- Define the list of tools ---
tools = [financial_data_tool, review_rag_tool, structured_sentiment_analyzer, web_scraper_tool]

# --- IMPROVED PROMPT TEMPLATE ---
# We've added more explicit instructions for the agent.
prompt = PromptTemplate.from_template("""
You are a master business analyst. Your job is to provide a comprehensive answer to the user's question.
You must use the tools available to you to gather information before forming your final answer.
Do not apologize if a tool fails; instead, state the result of the tool and continue with your plan.
Once you have gathered enough information from your tools, you must provide a final answer.

TOOLS:
------
{tools}

Here is the user's question:
{input}

Begin! Use the following format for your thought process:
Thought: I need to use a tool to find specific information.
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat)
Thought: I have now gathered all the information I need.
Final Answer: [Your final, synthesized answer to the user's original question]
Thought:{agent_scratchpad}
""")

# --- Create the Agent ---
agent = create_react_agent(llm, tools, prompt)

# --- IMPROVED AGENT EXECUTOR ---
# We've increased the max_iterations to give the agent more "thinking" time.
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10  # <-- INCREASED ITERATION LIMIT
)

@app.post("/ask_agent")
async def ask_master_agent(request: QueryRequest):
    """Receives a query and delegates it to the Master Agent."""
    if not request.query:
        raise HTTPException(status_code=400, detail="Query not provided")
    try:
        response = await agent_executor.ainvoke({"input": request.query})
        return {"response": response.get("output")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")

@app.get("/")
def root():
    return {"message": "Multi-Agent System API is online."}