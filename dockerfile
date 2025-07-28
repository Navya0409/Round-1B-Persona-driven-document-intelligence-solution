FROM --platform=linux/amd64 python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data during build (not runtime)
RUN python -c "import nltk; nltk.download('punkt', download_dir='/usr/local/nltk_data'); nltk.download('stopwords', download_dir='/usr/local/nltk_data')"

# Set NLTK data path
ENV NLTK_DATA=/usr/local/nltk_data

# Copy application code
COPY main.py .

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Run the application
CMD ["python", "main.py"]
