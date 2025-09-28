import os
import sys
import json
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- CONFIGURATION ---
PRODUCT_NARRATIVES_PATH = os.path.join(project_root, 'data', 'product_narratives.json')
RETURN_NARRATIVES_PATH = os.path.join(project_root, 'data', 'return_narratives.json')
CHROMA_HOST = os.environ.get("CHROMA_HOST", "chroma")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", 8000))
EMBEDDING_MODEL_NAME = 'mixedbread-ai/mxbai-embed-large-v1'
PRODUCT_COLLECTION_NAME = "product_performance_narratives"
RETURN_COLLECTION_NAME = "return_detail_narratives"
BATCH_SIZE = 512 # A safe batch size for embedding and DB insertion

def process_collection(collection_name: str, file_path: str, client: chromadb.HttpClient, model: SentenceTransformer):
    """
    Processes and ingests data for a given collection in sequential, memory-safe batches.
    """
    try:
        print(f"\n--- Processing '{collection_name}' collection ---")
        try:
            client.delete_collection(name=collection_name)
            print(f"  - Deleted existing collection: '{collection_name}'")
        except Exception:
            print(f"  - Collection '{collection_name}' did not exist. Creating new.")
            
        collection = client.create_collection(name=collection_name)
        
        with open(file_path, 'r') as f:
            narratives = json.load(f)
        
        total_docs = len(narratives)
        print(f"  - Found {total_docs} documents. Processing in batches of {BATCH_SIZE}...")

        for i in tqdm(range(0, total_docs, BATCH_SIZE), desc=f"Ingesting '{collection_name}'"):
            batch = narratives[i:i + BATCH_SIZE]
            
            ids = [item['id'] for item in batch]
            documents = [item['text'] for item in batch]
            metadatas = [item['metadata'] for item in batch]

            # Embed only the current batch
            embeddings = model.encode(documents, show_progress_bar=False).tolist()
            
            # Add the current batch to the collection
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
        
        print(f"✅ Successfully ingested {total_docs} documents into '{collection_name}'.")
        
    except Exception as e:
        print(f"❌ ERROR during '{collection_name}' ingestion: {e}")
        import traceback
        traceback.print_exc()

def run():
    """Main function to ingest product and return narratives into ChromaDB."""
    print(f"--- 🚀 Starting Vector Ingestion Pipeline ---")
    
    try:
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        print(f"Connecting to ChromaDB host at {CHROMA_HOST}:{CHROMA_PORT}...")
        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        chroma_client.heartbeat()
        print("✅ Successfully connected to ChromaDB.")
    except Exception as e:
        print(f"❌ ERROR: Could not initialize models or connect to ChromaDB. Halting. Error: {e}")
        return

    # Process both narrative files
    process_collection(PRODUCT_COLLECTION_NAME, PRODUCT_NARRATIVES_PATH, chroma_client, embedding_model)
    process_collection(RETURN_COLLECTION_NAME, RETURN_NARRATIVES_PATH, chroma_client, embedding_model)
        
    print("\n🎉 --- Vector Ingestion Pipeline Finished Successfully --- 🎉")

if __name__ == "__main__":
    run()