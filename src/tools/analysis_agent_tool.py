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
    type: str = Field(description="The type of chart, e.g., 'bar_chart' or 'line_chart'.")
    title: str = Field(description="The title of the chart.")
    data: List[Dict[str, Any]] = Field(description="The data for the chart, a list of dictionaries.")

class TableData(BaseModel):
    headers: List[str] = Field(description="A list of strings for the table column headers.")
    rows: List[List[Any]] = Field(description="A list of lists, where each inner list represents a row of data.")
    title: str = Field(description="A descriptive title for the table.")

class AnalysisReport(BaseModel):
    analysis_summary: str = Field(description="A detailed, narrative summary of the key findings.")
    key_insights: List[str] = Field(description="A bulleted list of the most important insights.")
    actionable_recommendations: List[str] = Field(description="A list of specific business recommendations.")
    data_quality_concerns: List[str] = Field(description="A list of any potential data quality issues observed.")
    table_data: Optional[TableData] = Field(None, description="Structured data for a summary table.")
    chart_data: Optional[ChartData] = Field(None, description="Structured data to generate a chart.")

class DataAnalysisInput(BaseModel):
    """Input model for the Data Analysis tool."""
    user_query: str = Field(description="The original, natural language query from the user.")
    data: str = Field(description="The consolidated data from previous steps that needs to be analyzed.")

@tool(args_schema=DataAnalysisInput)
def data_analysis_tool(user_query: str, data: str) -> str:
    """
    Use this tool at the very end of a query to perform a detailed analysis and structure the final output.
    It returns a JSON object with a summary, insights, and data for a table or chart.
    """
    print("--- 🔬 DATA ANALYST AGENT: Performing structured analysis... ---")
    parser = JsonOutputParser(pydantic_object=AnalysisReport)
    
    analyst_prompt_template = """
    You are a world-class business analyst. Analyze the provided dataset in the context of the user's query and generate a comprehensive JSON report.

    **Original User Query:** {user_query}
    **Data to Analyze:** {data}

    **Analysis Objective:**
    1.  Create a concise `analysis_summary`.
    2.  Extract 3-5 `key_insights`.
    3.  Propose 3-5 `actionable_recommendations`.
    4.  List any `data_quality_concerns`.
    5.  **Data Presentation Mandate:** Based on the user's query, you MUST populate EITHER `table_data` OR `chart_data`.
        - If the query asks for a "list", "ranking", or "top N", populate `table_data`.
        - If the query asks for a "trend", "comparison", or "breakdown", populate `chart_data`.
        - Leave the unused field as null.

    **JSON Output Format Instructions:** {format_instructions}
    """
    prompt = ChatPromptTemplate.from_template(
        analyst_prompt_template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    llm = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL_NAME"),
        temperature=0.1,
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    analysis_chain = prompt | llm | parser
    
    print("--- [Analysis Tool] Invoking analysis chain... ---")
    response_dict = analysis_chain.invoke({"data": data, "user_query": user_query})
    
    response_json_string = json.dumps(response_dict)
    print("--- ✅ DATA ANALYST AGENT: Structured analysis complete. ---")
    return response_json_string