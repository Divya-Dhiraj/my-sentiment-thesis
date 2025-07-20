# src/tools/custom_tools.py
import os
import requests
import pandas as pd
import chromadb
import matplotlib.pyplot as plt
import json
import re
from datetime import datetime

from langchain.agents import tool
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
from tavily import TavilyClient

from functools import lru_cache
from chromadb.config import Settings
from sqlalchemy import text
from thefuzz import process
from ..database import engine

# --- Pre-load expensive models ONCE when the app starts ---
print("--- Pre-loading Sentence Transformer model ---")
EMBEDDING_MODEL = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')
print("--- Model pre-loaded successfully ---")

try:
    TAVILY_CLIENT = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
except Exception as e:
    print(f"Warning: Tavily API key not found or invalid. Web search tool may not work. Error: {e}")
    TAVILY_CLIENT = None

def _extract_json_from_response(response_text: str) -> str:
    """Finds and extracts the first valid JSON object from a string."""
    match = re.search(r'```json\s*(\{.*?\})\s*```|(\{.*?\})', response_text, re.DOTALL)
    if match:
        return match.group(1) if match.group(1) else match.group(2)
    return response_text

# --- THIS IS THE UPGRADED "GROUNDED" HELPER FUNCTION ---
def get_dates_from_query(query: str) -> dict:
    """
    Uses an LLM to extract a start and end date from a natural language query,
    grounded by the actual date range in the database.
    """
    # 1. Find the actual date range from the database to "ground" the LLM
    try:
        with engine.connect() as conn:
            date_range_query = "SELECT MIN(week_start_date) as min_date, MAX(week_start_date) as max_date FROM weekly_performance;"
            result = conn.execute(text(date_range_query)).fetchone()
            min_db_date = result.min_date.strftime('%Y-%m-%d') if result and result.min_date else "2024-09-15"
            max_db_date = result.max_date.strftime('%Y-%m-%d') if result and result.max_date else "2025-09-15"
    except Exception as db_exc:
        print(f"Warning: Could not query DB for date range. Using default dates. Error: {db_exc}")
        min_db_date, max_db_date = "2024-09-15", "2025-09-15"

    # 2. Use the real date range to create a fact-based prompt
    date_parser_llm = ChatOpenAI(
        model="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
        api_key=os.environ.get("NOVITA_API_KEY"),
        base_url=os.environ.get("NOVITA_API_BASE_URL"),
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    
    prompt = f"""
    Based on the user's query, extract a start_date and end_date in 'YYYY-MM-DD' format.
    The data available in the database ranges from {min_db_date} to {max_db_date}.
    The current date is {max_db_date}.

    RULES:
    - If the user asks for the "entire duration", "all time", "from the beginning", or "since launch", use the full available date range: start_date should be {min_db_date} and end_date should be {max_db_date}.
    - If the user asks for a specific period like "last 4 months" or "in June 2025", calculate that based on the current date.
    - If no date is mentioned, default to the last 3 months.

    Query: "{query}"
    Respond ONLY with a single, valid JSON object with "start_date" and "end_date" keys.
    """
    try:
        response = date_parser_llm.invoke(prompt)
        clean_json_str = _extract_json_from_response(response.content)
        dates = json.loads(clean_json_str)
        print(f"--- Date Parser LLM extracted: {dates} ---")
        return dates
    except Exception as e:
        print(f"Warning: Date parsing failed. Using full date range as fallback. Error: {e}")
        return {"start_date": min_db_date, "end_date": max_db_date}


@tool
@lru_cache(maxsize=128)
def review_rag_tool(query: str) -> str:
    """Finds and returns the most relevant customer reviews for a given query."""
    print(f"--- RAGAgent: Finding reviews relevant to: '{query}' ---")
    try:
        chroma_client = chromadb.HttpClient(host=os.environ.get("CHROMA_HOST"), port=int(os.environ.get("CHROMA_PORT")), settings=Settings(anonymized_telemetry=False))
        collection = chroma_client.get_collection(name="product_reviews")
        query_embedding = EMBEDDING_MODEL.encode([query]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=3)
        if not results or not results.get('documents') or not results['documents'][0]:
            return "No relevant reviews found in the knowledge base."
        return "Found relevant reviews:\n- " + "\n- ".join(results['documents'][0])
    except Exception as e:
        return f"Error retrieving reviews from knowledge base: {e}"

@tool
@lru_cache(maxsize=128)
def financial_data_tool(query: str) -> str:
    """
    Returns weekly performance data for a product within a specified date range.
    Uses fuzzy matching to find products even with typos.
    """
    print(f"--- PerformanceAgent: Received query: {query} ---")
    try:
        products_df = pd.read_sql_query("SELECT asin, product_name FROM products;", engine)
        product_choices = products_df['product_name'].tolist()
        if not product_choices:
            return "Error: No products found in the database. Please run the ingestion script."
        best_match = process.extractOne(query, product_choices)
        product_name_found, product_id_found = None, None
        
        if best_match and best_match[1] >= 80:
            product_name_found = best_match[0]
            product_id_found = products_df[products_df['product_name'] == product_name_found]['asin'].iloc[0]
        else:
            return "Sorry, I couldn't find any relevant data on this product from internal sources. I will get some information in general from the LLM database."

        dates = get_dates_from_query(query)
        start_date, end_date = dates['start_date'], dates['end_date']
        print(f"--- PerformanceAgent: Matched '{query}' to '{product_name_found}' (ASIN: {product_id_found}), Date Range: {start_date} to {end_date} ---")

        with engine.connect() as conn:
            sql_query = """
                SELECT week_start_date, total_units_sold, average_selling_price, num_reviews_received, average_rating_new 
                FROM weekly_performance 
                WHERE asin = :asin AND week_start_date BETWEEN :start_date AND :end_date
                ORDER BY week_start_date;
            """
            params = {'asin': product_id_found, 'start_date': start_date, 'end_date': end_date}
            data = pd.read_sql_query(text(sql_query), conn, params=params)
        
        if data.empty:
            return f"No performance data found for {product_name_found} in the specified period ({start_date} to {end_date})."
        
        return f"Product Name: {product_name_found}\nPerformance data for {product_id_found} from {start_date} to {end_date}:\n{data.to_markdown(index=False)}"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error fetching performance data: {e}"

@tool
def plot_sales_trend(query: str) -> str:
    """Generates a sales trend plot. Uses fuzzy matching to find products."""
    print(f"--- PlottingAgent: Received query: {query} ---")
    try:
        products_df = pd.read_sql_query("SELECT asin, product_name FROM products;", engine)
        product_choices = products_df['product_name'].tolist()
        if not product_choices:
            return "Error: No products found in the database to plot."
        best_match = process.extractOne(query, product_choices)
        product_id_found, product_name_found = None, None
        
        if best_match and best_match[1] >= 80:
            product_name_found = best_match[0]
            product_id_found = products_df[products_df['product_name'] == product_name_found]['asin'].iloc[0]
        else:
            return "Sorry, I couldn't find a matching product in the internal database to create a plot."
            
        print(f"--- PlottingAgent: Generating sales plot for {product_id_found} ---")
        dates = get_dates_from_query(query)
        start_date, end_date = dates['start_date'], dates['end_date']
        
        with engine.connect() as conn:
            sql_query = """
                SELECT week_start_date, total_units_sold FROM weekly_performance 
                WHERE asin = :asin AND week_start_date BETWEEN :start_date AND :end_date 
                ORDER BY week_start_date;
            """
            params = {'asin': product_id_found, 'start_date': start_date, 'end_date': end_date}
            data = pd.read_sql_query(text(sql_query), conn, params=params)
            
        if data.empty:
            return f"No data to plot for {product_id_found} in the period {start_date} to {end_date}."

        plt.figure(figsize=(10, 6))
        plt.plot(pd.to_datetime(data['week_start_date']), data['total_units_sold'], marker='o')
        plt.title(f'Sales Trend for {product_name_found}')
        plt.xlabel('Week')
        plt.ylabel('Total Units Sold')
        plt.grid(True)
        plt.tight_layout()
        
        plot_filename = f"{product_id_found}_sales_trend.png"
        plt.savefig(f"/app/{plot_filename}")
        
        return f"Successfully generated and saved plot as {plot_filename} covering the period from {start_date} to {end_date}."
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error generating plot: {e}"

@tool
def structured_sentiment_analyzer(review_text: str) -> str:
    """Analyzes a product review and returns a structured JSON object with sentiment and aspects."""
    print(f"--- SentimentAgent: Analyzing text: '{review_text[:50]}...' ---")
    try:
        llm = ChatOpenAI(
            model="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
            api_key=os.environ.get("NOVITA_API_KEY"),
            base_url=os.environ.get("NOVITA_API_BASE_URL"),
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        prompt = f"""
        Analyze the sentiment of the following product review.
        Return ONLY a single, valid JSON object with two keys:
        1. "sentiment": a string, which must be one of ["Positive", "Negative", "Neutral"].
        2. "aspects": a Python list of specific features mentioned.
        Review: "{review_text}"
        """
        response = llm.invoke(prompt)
        clean_json_str = _extract_json_from_response(response.content)
        return clean_json_str
    except Exception as e:
        return f"Error during sentiment analysis: {e}"

@tool
def web_search_tool(query: str) -> str:
    """Performs a general web search and returns snippets from top results."""
    print(f"--- WebSearchAgent: Searching for: '{query}' ---")
    if not TAVILY_CLIENT:
        return "Web search tool is not configured. Please provide a valid TAVILY_API_KEY in .env."
    try:
        results = TAVILY_CLIENT.search(query=query, search_depth="basic", max_results=5)
        if not results or not results['results']:
            return "No relevant search results found."
        formatted_results = []
        for i, res in enumerate(results['results']):
            content = res.get('content', 'No content available.')
            if content is None: content = 'No content available.'
            formatted_results.append(
                f"Result {i+1}:\n"
                f"  Title: {res.get('title', 'N/A')}\n"
                f"  URL: {res.get('url', 'N/A')}\n"
                f"  Content: {content[:300]}...\n"
            )
        return "Found web search results:\n" + "\n".join(formatted_results)
    except Exception as e:
        return f"Error during web search: {e}"

@tool
def browse_website_tool(url: str) -> str:
    """Reads the content of a webpage at a given URL and returns its title."""
    print(f"--- WebBrowserAgent: Accessing URL: {url} ---")
    try:
        scraper_url = f"http://thesis_scraper:8003/scrape?url={url}"
        response = requests.get(scraper_url, timeout=15)
        response.raise_for_status()
        return f"Successfully read the webpage at {url}. The title is: '{response.json().get('title', 'N/A').strip()}'"
    except Exception as e:
        return f"Error accessing website: {e}"

@tool
def extract_web_data_tool(url_and_selector: str) -> str:
    """Extracts specific data from a webpage using a CSS selector."""
    print(f"--- DataExtractionAgent: Attempting to extract from: '{url_and_selector}' ---")
    try:
        parts = [p.strip() for p in url_and_selector.split('|')]
        if len(parts) != 2:
            return "Invalid input format. Expected 'URL | CSS_SELECTOR'."
        url, css_selector = parts[0], parts[1]
        scraper_url = f"http://thesis_scraper:8003/scrape?url={url}&css_selector={requests.utils.quote(css_selector)}"
        response = requests.get(scraper_url, timeout=15)
        response.raise_for_status()
        result_json = response.json()
        extracted_content = result_json.get('extracted_content', 'No content found.')
        if extracted_content.startswith("No element found"):
            return f"No data found at {url} for selector '{css_selector}'. The tool received: {extracted_content}"
        return f"Successfully extracted from {url} using selector '{css_selector}': {extracted_content}"
    except Exception as e:
        return f"Error extracting data: {e}"