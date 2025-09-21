# src/tools/analysis_agent_tool.py
import os
import json
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain.agents import tool
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class ChartData(BaseModel):
    type: str = Field(description="The type of chart, e.g., 'bar_chart', 'line_chart', or 'pie_chart'.")
    title: str = Field(description="The title of the chart.")
    x_axis_label: str = Field(description="The label for the X-axis.")
    y_axis_label: str = Field(description="The label for the Y-axis.")
    data: List[Dict[str, Any]] = Field(description="The data for the chart, typically a list of dictionaries.")

# --- NEW, UPGRADED ANALYSIS REPORT STRUCTURE ---
class AnalysisReport(BaseModel):
    analysis_summary: str = Field(description="A detailed, narrative summary of the key findings from the data.")
    key_insights: List[str] = Field(description="A bulleted list of 2-4 of the most important, non-obvious insights discovered in the data.")
    actionable_recommendations: List[str] = Field(description="A list of 2-3 specific, actionable business recommendations based on the insights.")
    data_quality_concerns: List[str] = Field(description="A bulleted list of any potential data quality issues, anomalies, or limitations observed (e.g., low sample size, missing data, outliers). If none, return an empty list.")
    chart_data: ChartData = Field(description="Structured data to generate a single, compelling chart that visualizes the key findings.")

class DataAnalysisInput(BaseModel):
    """Input model for the Data Analysis tool."""
    data: str = Field(description="The consolidated, pre-processed, and summarized data from previous steps that needs to be analyzed.")

@tool
def data_analysis_tool(tool_input: DataAnalysisInput) -> str:
    """
    Use this tool at the very end of a query to perform a detailed, structured analysis on the summarized data.
    It takes clean data and returns a rich JSON object containing a narrative summary, key insights, recommendations, data quality notes, and chart data.
    """
    data = tool_input.data
    print("--- 🔬 DATA ANALYST AGENT: Performing structured analysis... ---")
    parser = JsonOutputParser(pydantic_object=AnalysisReport)

    # --- NEW, UPGRADED PROMPT FOR THE ANALYSIS TOOL ---
    analyst_prompt_template = """
    You are a world-class, expert financial and business analyst working for a major e-commerce company.
    Your task is to analyze the provided dataset and generate a comprehensive, structured report in JSON format.
    You must adhere strictly to the JSON schema provided.

    **Analysis Objective:**
    1.  **Summarize:** Create a clear, narrative summary of what the data shows.
    2.  **Distill Insights:** Extract the most critical, non-obvious insights that a business leader would need to know.
    3.  **Recommend Actions:** Propose specific, actionable recommendations based on your findings.
    4.  **Assess Quality:** Identify any potential issues or limitations with the underlying data that could affect the conclusions.
    5.  **Visualize:** Structure the most important data points for a single, clear chart.

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
        stop=None, # Ensure compatibility
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    analysis_chain = prompt | llm | parser
    
    print("--- [Analysis Tool] Invoking analysis chain... ---")
    response_dict = analysis_chain.invoke({"data": data})
    
    # Add a debug print statement to see the rich JSON before it's returned
    print(f"--- [Analysis Tool] Generated raw analysis object: ---\n{json.dumps(response_dict, indent=2)}")

    response_json_string = json.dumps(response_dict, indent=2)
    print("--- ✅ DATA ANALYST AGENT: Structured analysis complete. ---")
    return response_json_string