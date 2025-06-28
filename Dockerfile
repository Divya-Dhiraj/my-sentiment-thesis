# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential libpq-dev

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')"

# --- ADD THIS LINE ---
# Create a data directory inside the container
RUN mkdir data

# Copy your project's source code into the container
COPY ./src /app/src
COPY run.py .

# Expose the port
EXPOSE 8001

# The default command
CMD ["python", "run.py", "start-api"]