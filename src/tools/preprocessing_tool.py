# src/tools/preprocessing_tool.py
import os
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.agents import tool
from pydantic import BaseModel, Field

class DataPreprocessorInput(BaseModel):
    """Input model for the Data Pre-processor tool."""
    data: str = Field(description="The raw, verbose data string from a previous tool's output that needs to be summarized.")

@tool
def data_preprocessing_tool(tool_input: DataPreprocessorInput) -> str:
    """
    Use this tool to summarize and clean raw data (like SQL results) into a concise, factual summary.
    """
    data = tool_input.data
    print("--- 🧹 DATA PRE-PROCESSOR: Cleaning and summarizing raw data... ---")

    preprocessor_prompt_template = """
    You are an efficient data pre-processing AI. Your sole job is to take raw, verbose data and extract the key facts.
    Do not analyze or interpret the data. Just summarize it factually and concisely.

    **Example 1:**
    - Raw Data: "[('B098K6C12Z',), ('B0CBRTRV1T',), ... 500 more items ...]"
    - Concise Summary: "Found 502 products of type DISHWASHER. The first few ASINs are B098K6C12Z, B0CBRTRV1T."

    **Example 2:**
    - Raw Data: "[('No reason given', 42)]"
    - Concise Summary: "The top return reason is 'No reason given' with a count of 42."
    
    **Your Task:**
    Summarize the following raw data. Be brief and factual.

    **Raw Data:**
    -------------------
    {data}
    -------------------

    **Concise Summary:**
    """
    
    prompt = ChatPromptTemplate.from_template(preprocessor_prompt_template)
    llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL_NAME"), temperature=0.0)
    preprocessing_chain = prompt | llm | StrOutputParser()
    
    response = preprocessing_chain.invoke({"data": data})
    print("--- ✅ DATA PRE-PROCESSOR: Data summarized successfully. ---")
    return response