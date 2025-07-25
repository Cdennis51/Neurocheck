# Base image
FROM python:3.10-slim

COPY neurocheck /neurocheck
COPY requirements.txt /requirements.txt
#COPY models

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn neurocheck.api_folder.api_file:app --host 0.0.0.0 --port $PORT
