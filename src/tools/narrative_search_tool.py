# src/tools/narrative_search_tool.py
import os
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.agents import tool
from pydantic import BaseModel, Field
from typing import Optional, List

# Load the embedding model once when the module is loaded
print("--- Loading embedding model for narrative search tool... ---")
EMBEDDING_MODEL = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')
print("--- Embedding model loaded. ---")

class NarrativeSearchInput(BaseModel):
    """Input model for the Narrative Search tool."""
    query: str = Field(description="The natural language query used for semantic search.")
    search_in: str = Field(description="The knowledge base to search in. Must be either 'products' or 'returns'.")
    asin_filter: Optional[List[str]] = Field(None, description="A list of ASINs to filter the search results.")

@tool(args_schema=NarrativeSearchInput)
def narrative_search_tool(query: str, search_in: str, asin_filter: Optional[List[str]] = None) -> str:
    """
    Use this tool to answer qualitative questions by searching through unstructured text narratives.
    It is ideal for questions about "why" something happened, summarizing themes, or finding specific examples.
    You MUST specify whether to search in the 'products' collection (for performance summaries) or the 'returns' collection (for specific return details).
    You can optionally filter by a list of ASINs.
    """
    print(f"--- 📖 Narrative Search Tool activated. Searching '{search_in}' for: '{query}' ---")
    
    if search_in not in ['products', 'returns']:
        return "Error: Invalid 'search_in' parameter. It must be either 'products' or 'returns'."

    collection_name = "product_performance_narratives" if search_in == 'products' else "return_detail_narratives"
    
    try:
        chroma_client = chromadb.HttpClient(host="chroma", port=8000)
        collection = chroma_client.get_collection(name=collection_name)
        
        query_embedding = EMBEDDING_MODEL.encode(query).tolist()
        
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": 5 # Return the top 5 most relevant narratives
        }
        
        if asin_filter:
            print(f"  - Applying ASIN filter: {asin_filter}")
            query_params["where"] = {"asin": {"$in": asin_filter}}
        
        results = collection.query(**query_params)
        
        documents = results.get('documents', [[]])[0]
        if not documents:
            return "No relevant narratives found for the given query and filters."
            
        formatted_results = "\n\n---\n\n".join(documents)
        print(f"--- ✅ Narrative search found {len(documents)} results. ---")
        return f"Found the following relevant narratives:\n{formatted_results}"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"An error occurred during narrative search: {e}"