FROM python:3.9.18-slim

WORKDIR /code

# Install dependencies
COPY ./server/requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./server /code/

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"] 