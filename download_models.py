# download_models.py
from sentence_transformers import SentenceTransformer

print("Downloading the embedding model. This will only happen once.")
SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1', trust_remote_code=True)
print("✅ Embedding model is downloaded and cached locally.")