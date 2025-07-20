# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential libpq-dev

# Copy ONLY the requirements file first to optimize caching
COPY requirements.txt .

# This layer will be cached as long as requirements.txt doesn't change
RUN pip install --no-cache-dir -r requirements.txt

# Create a data directory inside the container for any temporary files
RUN mkdir -p /app/data

# Copy your project's source code into the container
COPY ./src /app/src
COPY run.py .

# Expose the port the API will run on
EXPOSE 8001

# The default command to run when the container starts
CMD ["python", "run.py", "start-api"]