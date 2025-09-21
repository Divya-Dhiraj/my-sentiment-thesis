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
    data: List[Dict[str, Any]] = Field(description="The data for the chart, typically a list of dictionaries.")

class TableData(BaseModel):
    headers: List[str] = Field(description="A list of strings for the table column headers.")
    rows: List[List[Any]] = Field(description="A list of lists, where each inner list represents a row of data.")
    title: str = Field(description="The title of the table.")

class AnalysisReport(BaseModel):
    analysis_summary: str = Field(description="A detailed, narrative summary of the key findings from the data.")
    key_insights: List[str] = Field(description="A bulleted list of 2-4 of the most important, non-obvious insights discovered in the data.")
    actionable_recommendations: List[str] = Field(description="A list of 2-3 specific, actionable business recommendations based on the insights.")
    data_quality_concerns: List[str] = Field(description="A bulleted list of any potential data quality issues, anomalies, or limitations observed. If none, return an empty list.")
    table_data: Optional[TableData] = Field(None, description="Structured data for a summary table, to be used for rankings, lists, or detailed breakdowns.")
    chart_data: Optional[ChartData] = Field(None, description="Structured data to generate a single, compelling chart that visualizes the key findings.")

class DataAnalysisInput(BaseModel):
    """Input model for the Data Analysis tool."""
    data: str = Field(description="The consolidated, pre-processed, and summarized data from previous steps that needs to be analyzed.")

@tool
def data_analysis_tool(tool_input: DataAnalysisInput) -> str:
    """
    Use this tool at the very end of a query to perform a detailed, structured analysis on the summarized data.
    It returns a rich JSON object containing a narrative summary, insights, recommendations, and EITHER a table OR a chart when appropriate.
    """
    data = tool_input.data
    print("--- 🔬 DATA ANALYST AGENT: Performing structured analysis... ---")
    parser = JsonOutputParser(pydantic_object=AnalysisReport)

    # --- UPGRADED PROMPT with DATA PRESENTATION MANDATE ---
    analyst_prompt_template = """
    You are a world-class, expert financial and business analyst. Your task is to analyze the provided dataset and generate a comprehensive, structured report in JSON format.

    **Analysis Objective:**
    1.  Summarize the key findings in a narrative `analysis_summary`.
    2.  Distill the most critical takeaways into a `key_insights` list.
    3.  Propose specific `actionable_recommendations`.
    4.  Identify any `data_quality_concerns`.
    5.  **Data Presentation Mandate:** Based on the user's query and the data, you MUST decide on the best way to present the core data.
        - If the query asks for a "list", "ranking", "top N", or a detailed breakdown with multiple columns, you **MUST** populate the `table_data` field. Give it a clear `title`.
        - If the query asks for a "trend", "comparison between categories", "breakdown by a single dimension", or "share", you **MUST** populate the `chart_data` field. Choose the appropriate chart type (`bar_chart`, `line_chart`).
        - You should generally provide EITHER a table OR a chart, but not both. Choose the one that communicates the main point most effectively. If no visualization is needed, leave both fields null.

    **Dataset to Analyze:**
    -------------------
    {data}
    -------------------

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
    response_dict = analysis_chain.invoke({"data": data})
    
    print(f"--- [Analysis Tool] Generated raw analysis object: ---\n{json.dumps(response_dict, indent=2)}")

    response_json_string = json.dumps(response_dict, indent=2)
    print("--- ✅ DATA ANALYST AGENT: Structured analysis complete. ---")
    return response_json_string