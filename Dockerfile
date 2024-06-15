FROM python:3.10-slim

EXPOSE 8000

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Creates new user
RUN adduser -u 1000 --disabled-password --gecos "" appuser && chown -R appuser /app

# Install system dependencies
RUN apt update && apt full-upgrade -y && apt install -y gcc libpq-dev && apt clean

# Install pip requirements
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

COPY . /app

USER appuser

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]