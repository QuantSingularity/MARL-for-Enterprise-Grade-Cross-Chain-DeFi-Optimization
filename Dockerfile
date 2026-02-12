# Multi-Agent Reinforcement Learning - Production Docker Image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY code/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY code/ ./code/
COPY README.md .

# Set Python path
ENV PYTHONPATH=/app/code/src:$PYTHONPATH

# Create output directories
RUN mkdir -p /app/checkpoints /app/results /app/data /app/logs

# Default command
CMD ["python", "-m", "pytest", "code/tests/", "-v"]

# To run training:
# docker run -v $(pwd)/checkpoints:/app/checkpoints marl-defi python code/src/train/train_synthetic.py
