# src/utils.py
from langchain_community.utilities import SQLDatabase
from .database import engine
import chromadb
from sentence_transformers import SentenceTransformer

def get_db_schema() -> str:
    """Connects to the database and retrieves the schema of all tables."""
    print("--- 📚 Retrieving database schema... ---")
    db = SQLDatabase(engine)
    return db.get_table_info()

def check_vector_db_state():
    """
    Checks the state of the ChromaDB vector database to see if it needs ingestion.
    """
    print("--- 🔍 Checking Vector Database State... ---")
    
    # --- Step 1: Define the state of the CURRENT code ---
    # This is the model your application is currently configured to use.
    expected_model_name = 'mixedbread-ai/mxbai-embed-large-v1'
    try:
        model = SentenceTransformer(expected_model_name)
        expected_dimension = model.get_sentence_embedding_dimension()
        print(f"✅ Code expects embeddings with dimension: {expected_dimension}")
    except Exception as e:
        print(f"❌ Could not load the expected sentence transformer model: {e}")
        return

    # --- Step 2: Check the state of the LIVE database ---
    try:
        chroma_client = chromadb.HttpClient(host="chroma", port=8000)
        # Check if the client can connect to the server
        chroma_client.heartbeat() 
        print("✅ Successfully connected to ChromaDB server.")
        
        collection = chroma_client.get_collection(name="products")
        print(f"✅ Found collection 'products' with {collection.count()} items.")
        
        # Get one item to inspect its embedding dimension
        one_item = collection.peek(limit=1)
        if not one_item or not one_item.get('embeddings'):
             print("❌ Collection exists but is empty. Ingestion is required.")
             return

        actual_dimension = len(one_item['embeddings'][0])
        print(f"✅ Data in database has embedding dimension: {actual_dimension}")
        
        # --- Step 3: Compare and Conclude ---
        if actual_dimension == expected_dimension:
            print("\n--- ✅ STATUS: VectorDB is UP-TO-DATE. No ingestion needed. ---")
        else:
            print(f"\n--- ❌ STATUS: VectorDB is OUTDATED. Mismatch in embedding dimensions (Expected: {expected_dimension}, Found: {actual_dimension}).")
            print("---    Ingestion is REQUIRED. ---")

    except ValueError as e:
        # This specific error is often raised if the collection does not exist
        print(f"❌ Collection 'products' does not exist. Ingestion is required.")
    except Exception as e:
        print(f"❌ An error occurred while checking ChromaDB: {e}")
        print("---    Assuming ingestion is required. ---")