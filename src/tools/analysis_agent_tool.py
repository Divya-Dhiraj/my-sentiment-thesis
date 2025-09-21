# src/tools/analysis_agent_tool.py
import os
import json
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain.agents import tool
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ChartData(BaseModel):
    type: str = Field(description="The type of chart, e.g., 'bar_chart', 'line_chart', or 'pie_chart'.")
    title: str = Field(description="The title of the chart.")
    x_axis_label: str = Field(description="The label for the X-axis.")
    y_axis_label: str = Field(description="The label for the Y-axis.")
    data: List[Dict[str, Any]] = Field(description="The data for the chart, typically a list of dictionaries with keys like 'label' and 'value' or 'date' and 'value'.")

class TableData(BaseModel):
    headers: List[str] = Field(description="A list of strings for the table column headers.")
    rows: List[List[Any]] = Field(description="A list of lists, where each inner list represents a row of data.")
    title: str = Field(description="A descriptive title for the table.")

class AnalysisReport(BaseModel):
    analysis_summary: str = Field(description="A detailed, narrative summary of the key findings from the data.")
    key_insights: List[str] = Field(description="A bulleted list of 2-4 of the most important, non-obvious insights discovered in the data.")
    actionable_recommendations: List[str] = Field(description="A list of 2-3 specific, actionable business recommendations based on the insights.")
    data_quality_concerns: List[str] = Field(description="A bulleted list of any potential data quality issues, anomalies, or limitations observed. If none, return an empty list.")
    table_data: Optional[TableData] = Field(None, description="Structured data for a summary table, to be used for rankings, lists, or detailed breakdowns.")
    chart_data: Optional[ChartData] = Field(None, description="Structured data to generate a single, compelling chart that visualizes the key findings.")

# --- THIS IS THE FIX ---
# The tool's input now includes the user's original query for context.
class DataAnalysisInput(BaseModel):
    """Input model for the Data Analysis tool."""
    user_query: str = Field(description="The original, natural language query from the user.")
    data: str = Field(description="The consolidated, pre-processed, and summarized data from previous steps that needs to be analyzed.")

@tool(args_schema=DataAnalysisInput)
def data_analysis_tool(user_query: str, data: str) -> str:
    """
    Use this tool at the very end of a query to perform a detailed, structured analysis on the summarized data.
    It returns a rich JSON object containing a narrative summary, insights, recommendations, and EITHER a table OR a chart when appropriate.
    """
    print("--- 🔬 DATA ANALYST AGENT: Performing structured analysis... ---")
    parser = JsonOutputParser(pydantic_object=AnalysisReport)

    analyst_prompt_template = """
    You are a world-class, expert financial and business analyst. Your task is to analyze the provided dataset in the context of the original user query and generate a comprehensive, structured report in JSON format.

    **Original User Query:**
    {user_query}
    -------------------

    **Dataset to Analyze:**
    {data}
    -------------------

    **Analysis Objective:**
    1.  Read the user's original query and the accompanying data.
    2.  Synthesize the information to produce a concise `analysis_summary`.
    3.  Extract 3-5 `key_insights` that are not immediately obvious from the raw data.
    4.  Propose 3-5 concrete `actionable_recommendations` based on the insights.
    5.  List any `data_quality_concerns` you identify. If none, return an empty list.
    6.  **Data Presentation Mandate:** Based on the user's query and the data, you MUST decide on the best way to present the core data.
        - If the query asks for a "list", "ranking", "top N", or a detailed breakdown, you **MUST** populate the `table_data` field.
        - If the query asks for a "trend", "comparison", "breakdown by", or "share", you **MUST** populate the `chart_data` field.
        - You should generally provide EITHER a table OR a chart, not both. Choose the one that communicates the main point most effectively. If no visualization is needed, leave both fields null.

    **JSON Output Format Instructions:**
    {format_instructions}
    """
    prompt = ChatPromptTemplate.from_template(
        analyst_prompt_template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    llm = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL_NAME"),
        temperature=0.1,
        stop=None,
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    analysis_chain = prompt | llm | parser
    
    print("--- [Analysis Tool] Invoking analysis chain... ---")
    response_dict = analysis_chain.invoke({"data": data, "user_query": user_query})
    
    print(f"--- [Analysis Tool] Generated raw analysis object: ---")
    
    # Pretty-print the JSON for easier debugging in the logs
    response_json_string = json.dumps(response_dict, indent=2)
    print(response_json_string)

    print("--- ✅ DATA ANALYST AGENT: Structured analysis complete. ---")
    return response_json_string