# # src/api/main.py
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import os
# import uuid
# from typing import Optional

# from langchain.agents import AgentExecutor, create_react_agent
# from langchain.prompts import PromptTemplate
# from langchain.memory import ConversationBufferWindowMemory
# from langchain_openai import ChatOpenAI

# # Import BOTH the SQL tool and the NEW semantic search tool
# from src.tools.custom_tools import query_business_database_tool
# from src.tools.semantic_search_tool import semantic_product_search

# class QueryRequest(BaseModel):
#     query: str
#     session_id: Optional[str] = None

# app = FastAPI(title="Business Intelligence RAG Agent")

# SESSION_MEMORY = {}

# def get_agent_for_session(session_id: str):
#     """Creates or retrieves an agent executor for a given session ID."""
#     if session_id not in SESSION_MEMORY:
#         SESSION_MEMORY[session_id] = ConversationBufferWindowMemory(
#             k=5, memory_key="chat_history", return_messages=True
#         )
#     memory = SESSION_MEMORY[session_id]

#     llm = ChatOpenAI(
#         model=os.environ.get("NOVITA_MODEL_NAME"),
#         api_key=os.environ.get("NOVITA_API_KEY"),
#         base_url=os.environ.get("NOVITA_API_BASE_URL"),
#         temperature=float(os.environ.get("NOVITA_TEMPERATURE", 0.0)),
#         model_kwargs={"response_format": {"type": "text"}},
#     )
    
#     tools = [semantic_product_search, query_business_database_tool]

#     # This prompt includes the required {tool_names} variable
#     prompt = PromptTemplate.from_template("""
#     You are a powerful and precise Senior Business Intelligence Analyst.
#     Your goal is to answer user questions about product performance by using your available tools.

#     TOOLS:
#     ------
#     You have access to the following tools:
#     {tools}

#     RESPONSE FORMAT AND WORKFLOW:
#     --------------------
#     1.  **Analyze the User's Query**: Is the query specific (mentions an ASIN or exact product name) or is it vague/descriptive (e.g., "appliances for washing clothes")?

#     2.  **For Vague/Descriptive Queries**:
#         - You MUST first use the `semantic_product_search` tool to find relevant product ASINs.
#         - Once you have the ASINs, you MUST then use the `query_business_database_tool` with those ASINs to get performance data.

#     3.  **For Specific Queries**:
#         - If the user provides an ASIN or a very specific product name, use the `query_business_database_tool` directly.

#     4.  **Use the following format for your thought process**:

#         Thought: [Your reasoning on which tool to use and why]
#         Action: The action to take, which must be one of [{tool_names}]
#         Action Input: [The input for the chosen tool]
#         Observation: [The result of the action]

#     5.  **When you have the final answer, use this format**:

#         Thought: I now have the final answer.
#         Final Answer: [Your final, clear, and concise answer to the user]

#     Begin!

#     CONVERSATION HISTORY:
#     {chat_history}

#     Question: {input}
#     Thought:{agent_scratchpad}
#     """)
    
#     agent = create_react_agent(llm, tools, prompt)
    
#     agent_executor = AgentExecutor(
#         agent=agent,
#         tools=tools,
#         memory=memory,
#         verbose=True,
#         handle_parsing_errors=True,
#         max_iterations=10,
#     )
#     return agent_executor

# @app.post("/ask_agent")
# async def ask_agent(request: QueryRequest):
#     session_id = request.session_id or str(uuid.uuid4())
#     if not request.query:
#         raise HTTPException(status_code=400, detail="Query not provided")
    
#     try:
#         agent_executor = get_agent_for_session(session_id)
#         response = await agent_executor.ainvoke({"input": request.query})
#         return {"response": response.get("output"), "session_id": session_id}
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")

# @app.get("/")
# def root():
#     return {"message": "Business Intelligence RAG Agent API is online."}


# src/api/main.py
import asyncio
import re
import json
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
import os
import uuid
from typing import Optional, List, Dict, Any

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from src.tools.sql_agent_tool import sql_query_tool
from src.tools.semantic_search_tool import semantic_product_search
from src.tools.analysis_agent_tool import data_analysis_tool
from src.utils import get_db_schema

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

app = FastAPI(title="Business Intelligence RAG Agent")


# --- THE NEW "MASTER PLANNER" (V4 - Final Version) ---
def create_plan(query: str, db_schema: str, llm: ChatOpenAI) -> List[Dict]:
    print("--- 🧠 MASTER PLANNER: Decomposing the user's question... ---")
    
    planner_prompt_template = """
    You are a world-class business intelligence analyst and expert AI planner.
    Your primary role is to deconstruct a user's complex question into a logical, step-by-step plan that a downstream system of agents can execute.

    **Goal:** Create a JSON plan to answer the user's question comprehensively.

    **Available Tools:**
    1. `semantic_product_search(query: str)`: Use for vague, descriptive, or category-based questions about products. Finds product ASINs.
    2. `SQL Query Tool(question: str)`: Use to query the database for specific, factual data like sales, return reasons, prices, etc.
    3. `data_analysis_tool(data: str)`: Use this tool AFTER you have gathered all the necessary data. It performs calculations, summarizes findings, and provides business recommendations.

    **Thinking Process:**
    1.  **Identify the Core Intent:** What is the user's ultimate goal?
    2.  **Information Gathering:** What data is needed? Do I need to use `semantic_product_search` first? Then `SQL Query Tool`?
    3.  **Analysis Step:** Once all the raw data is gathered, I MUST add a final step to call the `data_analysis_tool`. The input for this tool should be the single, generic placeholder "[Result of Previous Steps]".
    4.  **Dependency Management:** Structure the plan with placeholders like "[Result of Step 1]". For example, a SQL query might need an ASIN from a previous step, so its input would be: "SELECT * FROM products WHERE asin = [Result of Step 1]". Notice there are NO quotes around the placeholder.

    **Output Format:**
    You must respond with ONLY a valid JSON list of dictionaries. Each dictionary must have "tool_name" and "tool_input" keys.

    **Database Schema for Context:**
    ----------------
    {db_schema}
    ----------------

    **User's Question:**
    "{input}"

    **JSON Plan:**
    """
    prompt = ChatPromptTemplate.from_template(planner_prompt_template)
    planner_chain = prompt | llm | StrOutputParser()
    plan_string = planner_chain.invoke({"input": query, "db_schema": db_schema})

    try:
        json_match = re.search(r"\[.*\]", plan_string, re.DOTALL)
        if not json_match:
            raise json.JSONDecodeError("No JSON list found in the planner's output.", plan_string, 0)
        plan = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        print(f"--- ⚠️ Planner output was not valid JSON. Raw output: ---\n{plan_string}")
        raise ValueError("The Master Planner failed to generate a valid JSON plan.")
    
    print("--- ✅ MASTER PLANNER: Plan generated successfully! Now validating steps... ---")
    
    validated_plan = []
    for i, step in enumerate(plan):
        if "tool_name" not in step or "tool_input" not in step:
            print(f"--- ❌ VALIDATION FAILED: Step {i+1} is malformed. ---")
            print(f"--- Offending Step Content: {step} ---")
            raise ValueError(f"Planner generated a malformed step. Check the logs for details.")
        
        print(f"Step {i+1}: Tool -> {step['tool_name']}, Input -> '{step['tool_input']}'")
        validated_plan.append(step)
        
    return validated_plan

@app.post("/ask_agent", response_model=Dict[str, Any])
async def ask_agent(request: QueryRequest):
    session_id = request.session_id or str(uuid.uuid4())
    if not request.query:
        raise HTTPException(status_code=400, detail="Query not provided")

    # This top-level try-except block will catch any unhandled errors from tools.
    try:
        print(f"--- Received complex query for session {session_id}: '{request.query}' ---")
        
        db_schema = get_db_schema()

        llm = ChatOpenAI(
            model=os.environ.get("NOVITA_MODEL_NAME"),
            api_key=os.environ.get("NOVITA_API_KEY"),
            base_url=os.environ.get("NOVITA_API_BASE_URL"),
            temperature=0.0
        )

        # 1. PLAN
        plan = create_plan(request.query, db_schema, llm)

        # 2. EXECUTE
        available_tools = {
            "semantic_product_search": semantic_product_search,
            "SQL Query Tool": sql_query_tool,
            "data_analysis_tool": data_analysis_tool,
        }
        step_results = []
        
        for i, step in enumerate(plan):
            tool_name = step["tool_name"]
            tool_input = step["tool_input"]

            # --- Placeholder replacement logic is now outside the try-except block ---
            if tool_name == "data_analysis_tool":
                all_previous_results = "\n\n".join(
                    [f"Result of Step {k+1}:\n{res}" for k, res in enumerate(step_results)]
                )
                tool_input = all_previous_results
            else:
                for j, prev_result in enumerate(step_results):
                    placeholder = f"[Result of Step {j+1}]"
                    if placeholder in tool_input:
                        core_result = prev_result
                        match = re.search(r"\(\s*'([^']+)'.*\)", prev_result)
                        if match:
                            core_result = match.group(1)

                        if re.match(r"\s*SELECT", tool_input, re.IGNORECASE):
                            sql_ready_values = ", ".join([f"'{item.strip()}'" for item in core_result.split(',')])
                            tool_input = tool_input.replace(placeholder, sql_ready_values)
                        else:
                            tool_input = tool_input.replace(placeholder, core_result)

            print(f"\n--- Executing Step {i+1}: Using tool '{tool_name}' with input '{tool_input}' ---\n")

            if tool_name not in available_tools:
                raise ValueError(f"Planner requested an unknown tool: {tool_name}")

            tool_to_use = available_tools[tool_name]
            
            # --- The try-except block per step is removed to allow tools to self-correct ---
            response = await asyncio.to_thread(tool_to_use.invoke, tool_input)
            step_results.append(response)

        # 3. FINALIZE AND RETURN STRUCTURED RESPONSE
        print("--- Finalizing structured response... ---")
        final_result_str = step_results[-1]
        
        try:
            final_response_data = json.loads(final_result_str)
            print("--- ✅ Successfully parsed final analysis JSON. ---")
        except (json.JSONDecodeError, TypeError):
            print("--- ⚠️ Final step was not valid JSON. Returning as text. ---")
            final_response_data = {"analysis_summary": final_result_str}

        final_response_data["session_id"] = session_id
        return final_response_data

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")


@app.get("/")
def root():
    return {"message": "Business Intelligence RAG Agent API is online."}