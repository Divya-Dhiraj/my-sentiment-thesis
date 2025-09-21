# src/tools/spreadsheet_tool.py
import os
import uuid
import pandas as pd
import re
from langchain.agents import tool
from pydantic import BaseModel, Field
from typing import List, Any

SHARED_STATIC_DIR = "/app/static"
os.makedirs(SHARED_STATIC_DIR, exist_ok=True)

class TableData(BaseModel):
    """Input model for the structured table data to be converted to a spreadsheet."""
    headers: List[str] = Field(description="A list of strings for the table column headers.")
    rows: List[List[Any]] = Field(description="A list of lists, where each inner list represents a row of data.")
    title: str = Field(description="A descriptive title for the spreadsheet, used as the filename.")

@tool
def spreadsheet_tool(table_data: TableData) -> str:
    """
    Use this tool to create a downloadable Excel spreadsheet from structured table data.
    It takes table headers and rows, saves them to an .xlsx file, and returns a confirmation message with the filename.
    Use this when a user asks to "export", "download", or receive data in "Excel" or "CSV" format.
    """
    try:
        print(f"--- 📄 SPREADSHEET TOOL: Generating an Excel file for '{table_data.title}'... ---")
        df = pd.DataFrame(table_data.rows, columns=table_data.headers)
        
        safe_title = re.sub(r'[^\w\s-]', '', table_data.title).strip().replace(' ', '_')
        filename = f"{safe_title}_{uuid.uuid4()}.xlsx"
        filepath = os.path.join(SHARED_STATIC_DIR, filename)

        print(f"--- [Spreadsheet Tool] Saving Excel file to: {filepath} ---")
        df.to_excel(filepath, index=False)

        return f"SUCCESS: The spreadsheet '{filename}' has been generated and is ready for download."

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error creating spreadsheet: {e}"