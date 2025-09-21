# ./Dockerfile

# --- Stage 1: The "Builder" ---
# This stage installs dependencies into a virtual environment.
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies needed for building some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git

# Create a virtual environment
RUN python -m venv /opt/venv

# Activate the virtual environment and upgrade pip
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip

# Copy ONLY the requirements file
COPY requirements.txt .

# Install the Python dependencies into the virtual environment
RUN pip install --no-cache-dir -r requirements.txt


# --- Stage 2: The "Final Image" ---
# This stage creates the slim, final image for running the application.
FROM python:3.11-slim

# --- THIS IS THE FIX ---
# Install git in the FINAL image and then clean up the apt cache to reduce image size.
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the virtual environment from the "builder" stage.
COPY --from=builder /opt/venv /opt/venv

# Copy the application source code.
COPY ./src /app/src
COPY run.py .

# Activate the virtual environment for all subsequent commands
ENV PATH="/opt/venv/bin:$PATH"

# Expose the port the API will run on
EXPOSE 8001

# The command to run the application using python from the venv
CMD ["python", "run.py", "start-api"]