# src/tools/custom_tools.py

# --- All imports are now cleanly at the top ---
import os
import json
import requests
import pandas as pd
import psycopg2
import chromadb
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from langchain.agents import tool
from langchain_community.chat_models import ChatOllama
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from .. import database
from tavily import TavilyClient
from functools import lru_cache


# NEW: Initialize Tavily Client outside the function to avoid re-initializing on every call
# Ensure TAVILY_API_KEY is set in your .env file
try:
    TAVILY_CLIENT = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
except Exception as e:
    print(f"Warning: Tavily API key not found or invalid. Web search tool may not work. Error: {e}")
    TAVILY_CLIENT = None # Set to None if API key is missing/invalid
# --- Tool 1: RAG Tool ---
@tool
@lru_cache(maxsize=128) # Also cache the RAG results
def review_rag_tool(query: str) -> str:
    """Finds and returns the most relevant customer reviews for a given query. Use this to understand customer opinions and complaints."""
    print(f"--- RAGAgent: Finding reviews relevant to: '{query}' ---")
    try:
        chroma_client = chromadb.HttpClient(host=os.environ.get("CHROMA_HOST"), port=int(os.environ.get("CHROMA_PORT")), settings=Settings(anonymized_telemetry=False))
        embedding_model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')
        collection = chroma_client.get_collection(name="product_reviews")
        
        query_embedding = embedding_model.encode([query]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=3)
        
        if not results or not results.get('documents') or not results['documents'][0]:
            return "No relevant reviews found in the knowledge base."
        return "Found relevant reviews:\n- " + "\n- ".join(results['documents'][0])
    except Exception as e:
        return f"Error retrieving reviews from knowledge base: {e}"

# --- Tool 2: Financial Tool ---
@tool
@lru_cache(maxsize=128) # Cache the last 128 calls to this tool
def financial_data_tool(query: str) -> str:
    """Returns financial data for a SINGLE product ID mentioned in the query. The input should be a string containing a valid product ID."""
    print(f"--- FinancialAgent: Received query: {query} ---")
    conn = None
    try:
        conn = database.get_db_connection()
        products_df = pd.read_sql_query("SELECT product_id FROM products;", conn)
        valid_ids = products_df['product_id'].str.upper().tolist()

        product_id_found = None
        for pid in valid_ids:
            if pid in query.upper():
                product_id_found = pid
                break
        
        if not product_id_found:
            return f"Could not identify a valid Product ID in the query. Valid IDs are: {valid_ids}"

        print(f"--- FinancialAgent: Extracted Product ID: {product_id_found} ---")
        sql_query = "SELECT date, sales_units, price, supplier_cost FROM financials WHERE product_id = %s;"
        data = pd.read_sql_query(sql_query, conn, params=(product_id_found,))
        
        if data.empty:
            return f"No financial data found for Product ID {product_id_found}."
        return f"Financial data for {product_id_found}:\n{data.to_markdown()}"
    except Exception as e:
        return f"Error fetching financial data: {e}"
    finally:
        if conn:
            conn.close()

# --- Tool 3: Dedicated Sentiment Tool ---
@tool
def structured_sentiment_analyzer(review_text: str) -> str:
    """Analyzes a product review and returns a structured JSON object with sentiment and aspects. The input must be the full text of the review."""
    print(f"--- SentimentAgent: Analyzing text: '{review_text[:50]}...' ---")
    try:
        #llm = ChatOllama(model="my-phi3", base_url=os.environ.get("OLLAMA_BASE_URL"), format="json")
         # +++ ADD THIS +++
        llm = ChatOpenAI(
            model="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
            api_key=os.environ.get("NOVITA_API_KEY"),
            base_url=os.environ.get("NOVITA_BASE_URL"),
            model_kwargs={
                "response_format": {"type": "json_object"}
            }
        )

        prompt = f"""
        Analyze the sentiment of the following product review.
        Return ONLY a single, valid JSON object with two keys:
        1. "sentiment": a string, which must be one of ["Positive", "Negative", "Neutral"].
        2. "aspects": a Python list of specific features mentioned.
        Review: "{review_text}"
        """
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error during sentiment analysis: {e}"

# --- Tool 5: Plotting Tool ---
@tool
def plot_sales_trend(query: str) -> str:
    """Generates and saves a sales trend plot for a given product ID mentioned in the query. Use this when asked to 'plot', 'graph', or 'visualize' sales data."""
    print(f"--- PlottingAgent: Received query: {query} ---")
    conn = None
    try:
        conn = database.get_db_connection()
        products_df = pd.read_sql_query("SELECT product_id FROM products;", conn)
        valid_ids = products_df['product_id'].str.upper().tolist()

        product_id_found = None
        for pid in valid_ids:
            if pid in query.upper():
                product_id_found = pid
                break
        
        if not product_id_found:
            return f"Could not identify a valid Product ID in the query to plot. Valid IDs are: {valid_ids}"

        print(f"--- PlottingAgent: Generating sales plot for {product_id_found} ---")
        sql_query = "SELECT date, sales_units FROM financials WHERE product_id = %s ORDER BY date;"
        data = pd.read_sql_query(sql_query, conn, params=(product_id_found,))
        
        if data.empty:
            return f"No data to plot for Product ID {product_id_found}."

        plt.figure(figsize=(10, 6))
        plt.plot(pd.to_datetime(data['date']), data['sales_units'], marker='o')
        plt.title(f'Sales Trend for Product {product_id_found}')
        plt.xlabel('Date')
        plt.ylabel('Units Sold')
        plt.grid(True)
        plt.tight_layout()
        
        plot_filename = f"{product_id_found}_sales_trend.png"
        plt.savefig(f"/app/{plot_filename}")
        
        return f"Successfully generated and saved plot as {plot_filename}."
    except Exception as e:
        return f"Error generating plot: {e}"
    finally:
        if conn:
            conn.close()


# --- Tool 6: General Web Search Tool ---
@tool
def web_search_tool(query: str) -> str:
    """Performs a general web search and returns snippets from top results. Use this when you need to find information on the internet that is not in your databases, like current prices, news, or external product specifications. Input should be a concise search query."""
    print(f"--- WebSearchAgent: Searching for: '{query}' ---")
    if not TAVILY_CLIENT:
        return "Web search tool is not configured. Please provide a valid TAVILY_API_KEY in .env."
    try:
        # Perform the search
        results = TAVILY_CLIENT.search(query=query, search_depth="basic", max_results=5) # Can adjust search_depth and max_results
        
        if not results or not results['results']:
            return "No relevant search results found."
        
        # Format results into a readable string
        formatted_results = []
        for i, res in enumerate(results['results']):
            # Ensure 'content' key exists and is not None
            content = res.get('content', 'No content available.')
            if content is None:
                content = 'No content available.'

            formatted_results.append(
                f"Result {i+1}:\n"
                f"  Title: {res.get('title', 'N/A')}\n"
                f"  URL: {res.get('url', 'N/A')}\n"
                f"  Content: {content[:300]}...\n" # Truncate long content
            )
        
        return "Found web search results:\n" + "\n".join(formatted_results)
    except Exception as e:
        return f"Error during web search: {e}"


@tool
def browse_website_tool(url: str) -> str:
    """Reads the content of a webpage at a given URL and returns its title. Use this to access live, real-time information from a specific webpage."""
    print(f"--- WebBrowserAgent: Accessing URL: {url} ---")
    try:
        # This calls the *modified* scraper service
        scraper_url = f"http://thesis_scraper:8003/scrape?url={url}"
        response = requests.get(scraper_url, timeout=15)
        response.raise_for_status() # Raise an error for bad responses (4xx or 5xx)
        # Ensure 'title' key exists in the response from the scraper
        return f"Successfully read the webpage at {url}. The title is: '{response.json().get('title', 'N/A').strip()}'"
    except Exception as e:
        return f"Error accessing website: {e}"

# NEW TOOL: For specific data extraction
@tool
def extract_web_data_tool(url_and_selector: str) -> str:
    """
    Extracts specific data from a webpage using a CSS selector.
    Input should be a string in the format "URL | CSS_SELECTOR".
    Example: "https://www.bestbuy.com/iphone-16 | .price-display"
    Use this when you have a specific URL and know exactly what data you need (e.g., price, stock status).
    """
    print(f"--- DataExtractionAgent: Attempting to extract from: '{url_and_selector}' ---")
    try:
        parts = [p.strip() for p in url_and_selector.split('|')]
        if len(parts) != 2:
            return "Invalid input format. Expected 'URL | CSS_SELECTOR'."
        
        url = parts[0]
        css_selector = parts[1]

        # Call the enhanced web_scraper service with the selector
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