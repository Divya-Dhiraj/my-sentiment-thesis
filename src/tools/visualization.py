# src/tools/visualization_tool.py
import os
import uuid
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from langchain.agents import tool
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

SHARED_STATIC_DIR = "/app/static"
os.makedirs(SHARED_STATIC_DIR, exist_ok=True)

class ChartData(BaseModel):
    """Input model for the structured chart data."""
    type: str = Field(description="The type of chart, e.g., 'bar_chart' or 'line_chart'.")
    title: str = Field(description="The title of the chart.")
    data: List[Dict[str, Any]] = Field(description="The data for the chart, a list of dictionaries.")
    x_axis_label: Optional[str] = Field(None, description="The label for the X-axis.")
    y_axis_label: Optional[str] = Field(None, description="The label for the Y-axis.")

@tool
def visualization_tool(chart_data: ChartData) -> str:
    """
    Use this tool to create a visual chart from structured data.
    It takes a ChartData object, generates a plot, saves it as an image, and returns a markdown link to that image.
    This should only be used when a user explicitly asks for a 'graph', 'chart', 'plot', or 'visualization'.
    """
    try:
        print(f"--- 📊 VISUALIZATION TOOL: Generating a {chart_data.type}... ---")
        df = pd.DataFrame(chart_data.data)

        if df.empty:
            return "Error: No data provided to generate the chart."

        # Rename columns for clarity if they are generic like 'x' and 'y'
        if 'x' in df.columns and 'y' in df.columns:
            df = df.rename(columns={'x': 'label', 'y': 'value'})

        numeric_cols = [col for col in df.columns if col not in ['label', 'date']]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 6))

        plot_kind = 'bar' if chart_data.type == 'bar_chart' else 'line'
        x_col = 'label' if 'label' in df.columns else 'date'
        
        if x_col not in df.columns:
            return f"Error: Required x-axis column '{x_col}' not found in data."
            
        y_cols = [col for col in df.columns if col != x_col]
        
        if x_col == 'date':
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').set_index('date')
            df.plot(kind=plot_kind, y=y_cols, ax=ax, rot=45)
        else: # Bar chart
            df.plot(kind=plot_kind, x=x_col, y=y_cols, ax=ax, rot=45)

        if plot_kind == 'bar':
            plt.xticks(ha='right')
        
        ax.set_title(chart_data.title, fontsize=16, color='white', pad=20)
        ax.set_xlabel(chart_data.x_axis_label or x_col.replace('_', ' ').title(), color='white')
        ax.set_ylabel(chart_data.y_axis_label or y_cols[0].replace('_', ' ').title(), color='white')
        
        ax.tick_params(colors='white', which='both')
        for spine in ax.spines.values():
            spine.set_color('gray')
        
        ax.get_yaxis().set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.tight_layout()
        
        filename = f"chart_{uuid.uuid4()}.png"
        filepath = os.path.join(SHARED_STATIC_DIR, filename)
        
        print(f"--- [Vis Tool] Saving chart to: {filepath} ---")
        plt.savefig(filepath, bbox_inches='tight', transparent=True, dpi=150)
        plt.close(fig)

        return f"![{chart_data.title}](/static/{filename})"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error creating visualization: {e}"