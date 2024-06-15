FROM python:3.10-slim

EXPOSE 8000

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    swig \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Creates new user
RUN adduser -u 1000 --disabled-password --gecos "" appuser && chown -R appuser /app

# Copy requirements
COPY requirements.txt .

# Upgrade pip and install requirements with increased timeout
RUN pip install --upgrade pip && \
    pip install --default-timeout=2000 --no-cache-dir -r requirements.txt

COPY . /app

USER appuser

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "main:app"]