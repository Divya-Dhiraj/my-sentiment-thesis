# src/tools/analysis_agent_tool.py
import os
import json
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain.agents import tool
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# --- Define the structured output we want ---
class ChartData(BaseModel):
    type: str = Field(description="The type of chart, e.g., 'bar_chart', 'line_chart', or 'pie_chart'.")
    title: str = Field(description="The title of the chart.")
    x_axis_label: str = Field(description="The label for the X-axis.")
    y_axis_label: str = Field(description="The label for the Y-axis.")
    data: List[Dict[str, Any]] = Field(description="The data for the chart, typically a list of dictionaries like [{'label': 'A', 'value': 100}].")

class AnalysisReport(BaseModel):
    analysis_summary: str = Field(description="A textual summary of the key findings from the data.")
    actionable_recommendations: List[str] = Field(description="A list of 2-3 actionable business recommendations.")
    chart_data: ChartData = Field(description="The structured data required to generate a chart that visualizes the key findings.")

@tool
def data_analysis_tool(data: str) -> str:
    """
    Use this tool to perform detailed analysis on a given dataset.
    It takes a string containing the raw data and returns a structured JSON object
    containing a summary, recommendations, and data for a chart.
    """
    print("--- 🔬 DATA ANALYST AGENT: Performing structured analysis... ---")

    parser = JsonOutputParser(pydantic_object=AnalysisReport)

    analyst_prompt_template = """
    You are a world-class, expert financial and business analyst.
    You are given a dataset. Your goal is to provide a deep, insightful analysis and structure it as a JSON object.

    **Critical Rule:** If the provided "Data to Analyze" is empty, contains an error message, or clearly indicates that no data was found, you MUST still produce a valid JSON object. In this case, your `analysis_summary` should explain the problem (e.g., "No data was found for the specified query."). The `actionable_recommendations` should be an empty list, and the `chart_data` should contain default or empty values.

    Based on the data, you must generate a JSON object that conforms to the following schema.
    Do not include any other text or markdown formatting around the JSON object.

    **JSON Schema:**
    {format_instructions}

    **Data to Analyze:**
    -------------------
    {data}
    -------------------

    **JSON Output:**
    """
    
    prompt = ChatPromptTemplate.from_template(
        analyst_prompt_template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    llm = ChatOpenAI(
        model=os.environ.get("NOVITA_MODEL_NAME"),
        api_key=os.environ.get("NOVITA_API_KEY"),
        base_url=os.environ.get("NOVITA_API_BASE_URL"),
        temperature=0.1,
        model_kwargs={"response_format": {"type": "json_object"}} # Important for consistent JSON output
    )

    analysis_chain = prompt | llm | parser
    
    response_dict = analysis_chain.invoke({"data": data})
    
    # Convert the dictionary to a JSON string for consistent tool output
    response_json_string = json.dumps(response_dict, indent=2)
    
    print("--- ✅ DATA ANALYST AGENT: Structured analysis complete. ---")
    return response_json_string