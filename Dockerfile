# ./Dockerfile

# --- Stage 1: The "Builder" ---
# This stage installs dependencies into a virtual environment.
# It uses a full Python image that includes build tools.
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies needed for building some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends build-essential

# Create a virtual environment
RUN python -m venv /opt/venv

# Activate the virtual environment and upgrade pip
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip

# Copy ONLY the requirements file
COPY requirements.txt .

# Install the Python dependencies into the virtual environment
# This layer is cached and only re-runs if requirements.txt changes.
RUN pip install --no-cache-dir -r requirements.txt


# --- Stage 2: The "Final Image" ---
# This stage creates the slim, final image for running the application.
FROM python:3.11-slim

WORKDIR /app

# Copy the virtual environment from the "builder" stage.
# This is a very fast operation.
COPY --from=builder /opt/venv /opt/venv

# Copy the application source code.
# Changing your source code will only invalidate this layer and below.
COPY ./src /app/src
COPY run.py .

# Activate the virtual environment for all subsequent commands
ENV PATH="/opt/venv/bin:$PATH"

# Expose the port the API will run on
EXPOSE 8001

# The command to run the application using python from the venv
CMD ["python", "run.py", "start-api"]