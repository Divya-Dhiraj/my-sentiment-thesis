# src/tools/semantic_search_tool.py
import os
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.agents import tool
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

print("--- Loading embedding model for semantic search tool... ---")
EMBEDDING_MODEL = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')
print("--- Embedding model loaded. ---")

class SemanticSearchInput(BaseModel):
    """Input model for the Semantic Product Search tool."""
    query: str = Field(description="The natural language query for finding products.")
    category: Optional[str] = Field(None, description="Filter to a specific 'clean_category'.")
    max_price: Optional[float] = Field(None, description="Filter for products with a price less than this value.")
    min_sales: Optional[int] = Field(None, description="Filter for products with total sales greater than this value.")

def _generate_search_queries(original_query: str) -> List[str]:
    """Uses an LLM to generate a set of alternative search queries."""
    print("--- 🧠 Engaging Multi-Query Generator... ---")
    query_gen_prompt_template = """
    You are an AI assistant who is an expert at converting user questions into effective vector database search queries.
    For the given user question, generate 3 additional, different search queries that are likely to find relevant documents.
    The queries should be rephrased from different perspectives and include different keywords.
    Provide ONLY the queries, each on a new line. Do not number them or add any other text.

    Original Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(query_gen_prompt_template)
    llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL_NAME"), temperature=0.0)
    chain = prompt | llm | StrOutputParser()
    generated_queries_str = chain.invoke({"question": original_query})
    queries = [q for q in generated_queries_str.split('\n') if q]
    queries.insert(0, original_query)
    print(f"--- ✅ Generated {len(queries)} search queries: {queries} ---")
    return queries

@tool
def semantic_product_search(tool_input: SemanticSearchInput) -> str:
    """
    Use this tool for vague or descriptive queries about products to find relevant ASINs.
    It supports hybrid search with semantic queries and metadata filters.
    """
    query, category, max_price, min_sales = tool_input.query, tool_input.category, tool_input.max_price, tool_input.min_sales
    print(f"--- 🔎 Multi-Query Hybrid Search activated for query: '{query}' with filters: category={category}, max_price={max_price}, min_sales={min_sales} ---")
    
    try:
        search_queries = _generate_search_queries(query)
        where_clause: Dict[str, Any] = {}
        if category: where_clause['clean_category'] = category
        if max_price is not None: where_clause['average_price'] = {"$lte": max_price}
        if min_sales is not None: where_clause['total_units_sold'] = {"$gte": min_sales}
        if len(where_clause) > 1: where_clause = {"$and": [{k: v} for k, v in where_clause.items()]}
        print(f"--- ChromaDB Where Clause: {where_clause} ---")
        
        chroma_client = chromadb.HttpClient(host="chroma", port=8000)
        collection = chroma_client.get_collection(name="products")
        
        all_found_asins = set()
        for q in search_queries:
            print(f"---   - Sub-querying for: '{q}' ---")
            query_embedding = EMBEDDING_MODEL.encode(q).tolist()
            query_params = {"query_embeddings": [query_embedding], "n_results": 5}
            if where_clause: query_params["where"] = where_clause
            
            results = collection.query(**query_params)
            ids = results.get('ids', [[]])[0]
            if ids: all_found_asins.update(ids)

        if not all_found_asins:
            return "No relevant products found via hybrid search with the given filters."
        
        clean_output = ', '.join(all_found_asins)
        print(f"--- ✅ Multi-Query Hybrid Search found {len(all_found_asins)} unique ASINs: {clean_output} ---")
        return clean_output
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error during hybrid search: {e}"