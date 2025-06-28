# src/tools/custom_tools.py
from langchain.agents import tool
from langchain_community.chat_models import ChatOllama
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import psycopg2
import os
import pandas as pd
import json
from .. import database

# --- Tool 1: RAG Tool ---
@tool
def review_rag_tool(query: str) -> str:
    """
    Finds and returns the most relevant customer reviews for a given query.
    Use this to understand customer opinions and complaints.
    """
    print(f"--- RAGAgent: Finding reviews relevant to: '{query}' ---")
    try:
        chroma_client = chromadb.HttpClient(
            host=os.environ.get("CHROMA_HOST"),
            port=int(os.environ.get("CHROMA_PORT")),
            settings=Settings(anonymized_telemetry=False)
        )
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
def financial_data_tool(query: str) -> str:
    """
    Returns financial data (sales, price, cost) for a given product ID mentioned in the query.
    Use this to get concrete numbers on sales performance. The input should be a string containing the product ID.
    """
    print(f"--- FinancialAgent: Received query: {query} ---")

    # A simple but more robust way to find a product ID like 'A101' or 'B202'
    # In a real project, you might use regex or an NLP model for this (like another agent!)
    product_id = None
    for word in query.split():
        if word.upper() in ['A101', 'B202']: # List of known product IDs
            product_id = word.upper()
            break

    if not product_id:
        return f"Could not identify a valid Product ID in the query '{query}'. Please specify a product ID like 'A101' or 'B202'."

    print(f"--- FinancialAgent: Extracted Product ID: {product_id} ---")
    try:
        conn = database.get_db_connection()
        sql_query = "SELECT date, sales_units, price, supplier_cost FROM financials WHERE product_id = %s;"
        data = pd.read_sql_query(sql_query, conn, params=(product_id,))
        conn.close()
        if data.empty:
            return f"No financial data found for Product ID {product_id}."
        return f"Financial data for {product_id}:\n{data.to_markdown()}"
    except Exception as e:
        return f"Error fetching financial data: {e}"



# --- Tool 3: Dedicated Sentiment Tool (NEW) ---
@tool
def structured_sentiment_analyzer(review_text: str) -> str:
    """
    Analyzes a product review and returns a structured JSON object.
    Use this tool for detailed analysis of a single review's sentiment and key topics.
    The input must be the full text of the review.
    """
    print(f"--- SentimentAgent: Analyzing text: '{review_text[:50]}...' ---")
    try:
        # We create a new LLM instance here with the format="json" parameter
        # This tells the LLM to guarantee its output is valid JSON.
        llm = ChatOllama(
            model="my-phi3", 
            base_url=os.environ.get("OLLAMA_BASE_URL"),
            format="json" # Force JSON output
        )

        prompt = f"""
        You are a sentiment analysis expert. Analyze the following product review.
        Your response MUST be a single, valid JSON object with two keys:
        1. "sentiment": a string, which must be one of ["Positive", "Negative", "Neutral"].
        2. "aspects": a Python list of strings, identifying the key features mentioned.

        Review: "{review_text}"
        """
        response = llm.invoke(prompt)
        # The response.content will be a clean JSON string
        return response.content
    except Exception as e:
        return f"Error during sentiment analysis: {e}"


@tool
def web_scraper_tool(url: str) -> str:
    """
    Scrapes a given URL and returns the title of the web page.
    Use this to get information from live websites. The input must be a valid URL.
    """
    print(f"--- WebScraperAgent: Scraping URL: {url} ---")
    try:
        # The agent_app container calls the web_scraper container using its service name
        response = requests.get(f"http://web_scraper:8003/scrape?url={url}")
        response.raise_for_status()
        return str(response.json())
    except Exception as e:
        return f"Error scraping website: {e}"