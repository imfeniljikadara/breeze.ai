FROM python:3.9-slim

WORKDIR /code

# Create a non-root user
RUN useradd -m -u 1000 user && \
    chown -R user:user /code

# Set environment variables
ENV HOME=/home/user \
    TRANSFORMERS_CACHE=/home/user/cache \
    HF_HOME=/home/user/cache

# Install dependencies
COPY ./server/requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./server /code/

# Set ownership
RUN chown -R user:user /code

# Switch to non-root user
USER user

# Create cache directory with correct permissions
RUN mkdir -p /home/user/cache && \
    chmod 755 /home/user/cache

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"] 