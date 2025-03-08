FROM python:3.9-slim

WORKDIR /code

COPY ./server/requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./server /code/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"] 