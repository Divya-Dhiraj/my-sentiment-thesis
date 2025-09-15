# src/tools/semantic_search_tool.py
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.agents import tool

# Pre-load the model once when the application starts for efficiency
print("--- Loading embedding model for semantic search tool... ---")
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
print("--- Embedding model loaded. ---")

@tool
def semantic_product_search(query: str) -> str:
    """
    Use this tool FIRST for vague or descriptive queries about products to find relevant ASINs.
    For example: "appliances for washing clothes", "something to clean my dishwasher", "which filters do you sell?".
    It takes a natural language query and returns a comma-separated string of the most semantically similar product ASINs.
    """
    print(f"--- 🔎 Semantic Search Tool activated for query: '{query}' ---")
    try:
        # Connect to the ChromaDB service running in Docker
        chroma_client = chromadb.HttpClient(host="chroma", port=8000)
        collection = chroma_client.get_collection(name="products")

        # Generate an embedding for the user's query
        query_embedding = EMBEDDING_MODEL.encode(query).tolist()

        # Query the collection to find the 5 most similar products
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )
        
        # Extract the ASINs from the results
        ids = results.get('ids', [[]])[0]
        if not ids:
            return "No relevant products found via semantic search."
        
        # --- THE FIX: Return ONLY the comma-separated string of ASINs ---
        # This clean output is crucial for downstream tools like SQL queries.
        clean_output = ', '.join(ids)
        print(f"--- ✅ Semantic Search found ASINs: {clean_output} ---")
        return clean_output

    except Exception as e:
        return f"Error during semantic search: {e}"