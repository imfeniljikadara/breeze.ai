FROM python:3.9.18-slim

WORKDIR /code

# Set environment variables
ENV HF_HOME=/code/cache \
    PYTHONUNBUFFERED=1

# Install dependencies
COPY ./server/requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create cache directory with correct permissions
RUN mkdir -p /code/cache && \
    chmod -R 777 /code/cache

# Copy application code
COPY ./server /code/

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"] 