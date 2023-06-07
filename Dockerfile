FROM python:3.10-slim

EXPOSE 3000

ENV PYTHONDONTWRITEBYTECODE=1

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Creates new user
RUN adduser -u 1000 --disabled-password --gecos "" appuser && chown -R appuser /app

# Install deps
RUN apt update && apt full-upgrade -y && apt clean

# Install pip requirements
RUN pip install --no-cache-dir -U pip setuptools wheel
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

USER appuser

CMD ["uvicorn", "--host", "0.0.0.0", "main:app"]
