# Use the official Python slim image for a lightweight backend
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the API requirements file
COPY requirements-api.txt .

# Install system dependencies required for building dlib
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install the backend dependencies
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy the API script and the Firebase credentials
# IMPORTANT: When uploading to Hugging Face Spaces, you must securely upload your serviceAccountKey.json
# or use Hugging Face Secrets to pass the credentials via environment variables.
COPY api.py .
# COPY serviceAccountKey.json . 

# Hugging Face Spaces typically use port 7860
EXPOSE 7860

# Command to run the FastAPI server on port 7860
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
