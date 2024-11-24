# frontend/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the frontend code
COPY . .

EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "front.py", "--server.address", "0.0.0.0"]