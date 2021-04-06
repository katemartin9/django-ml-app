FROM python:3.8-slim

COPY requirements.txt /code/
WORKDIR /code
RUN pip install -r requirements.txt
COPY kML /code/
