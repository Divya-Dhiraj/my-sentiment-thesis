# src/tools/re_ranking_tool.py
import os
from langchain.agents import tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import List

class ReRankerInput(BaseModel):
    """Input model for the Re-ranker and Fusion tool."""
    original_query: str = Field(description="The user's original, natural language query.")
    semantic_results: str = Field(description="A comma-separated string of ASINs from the semantic_product_search tool.")
    sql_results: str = Field(description="A comma-separated string of ASINs from the sql_query_tool.")

@tool
def re_ranker_tool(tool_input: ReRankerInput) -> str:
    """
    Use this final tool to merge, de-duplicate, and re-rank product results from both semantic and SQL searches.
    It takes the original user query and the ASIN lists from both sources to produce a final, definitive list.
    """
    original_query = tool_input.original_query
    semantic_results = tool_input.semantic_results
    sql_results = tool_input.sql_results
    
    print("--- 🧠 RE-RANKER AGENT: Fusing and re-ranking results... ---")
    print(f"--- [Re-ranker] Original Query: {original_query}")
    print(f"--- [Re-ranker] Semantic Results: {semantic_results}")
    print(f"--- [Re-ranker] SQL Results: {sql_results}")

    re_ranker_prompt_template = """
    You are an intelligent re-ranking and fusion engine. Your task is to take two lists of product ASINs and merge them into a single, de-duplicated, and logically ranked list that best answers the user's original query.

    **User's Original Query:**
    "{original_query}"

    **Input Lists:**
    1.  **Semantic Search Results (broad relevance):** These products are conceptually similar to the user's query.
        - ASINs: {semantic_results}
    2.  **SQL Query Results (strict criteria):** These products meet specific, structured criteria mentioned in the query (e.g., high sales, specific category).
        - ASINs: {sql_results}

    **Your Instructions:**
    1.  **De-duplicate:** Combine both lists and remove any duplicate ASINs.
    2.  **Re-rank:** Analyze the user's original query. Prioritize items that appear in BOTH lists, as they satisfy both semantic and structured criteria. Then, append the remaining items from the SQL results, followed by the remaining items from the semantic results.
    3.  **Format Output:** Return a single, comma-separated string of the final, ranked ASINs. Do not include any other text, explanations, or formatting.

    **Example:**
    - Original Query: "popular water filters"
    - Semantic Results: "ASIN1, ASIN2, ASIN3"
    - SQL Results: "ASIN2, ASIN4"
    - Final Output: "ASIN2, ASIN4, ASIN1, ASIN3"

    **Final Ranked, Comma-Separated ASINs:**
    """

    prompt = ChatPromptTemplate.from_template(re_ranker_prompt_template)
    
    llm = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL_NAME"),
        temperature=0.0,
        stop=None
    )

    parser = StrOutputParser()
    re_ranking_chain = prompt | llm | parser

    final_list = re_ranking_chain.invoke({
        "original_query": original_query,
        "semantic_results": semantic_results,
        "sql_results": sql_results
    })
    
    print(f"--- ✅ RE-RANKER AGENT: Final ranked list: {final_list} ---")
    return final_list