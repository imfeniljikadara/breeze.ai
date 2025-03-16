FROM python:3.9

WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY ./requirements.txt /code/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application
COPY . /code

# Create cache directory with proper permissions
RUN mkdir -p /cache && chmod 777 /cache

# Set environment variables
ENV TRANSFORMERS_CACHE=/cache
ENV HF_HOME=/cache

# Expose the port
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860"] 