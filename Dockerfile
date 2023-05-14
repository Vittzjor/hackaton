FROM python:3.9.0-slim

ENV PYTHONUNBUFFERED 1

EXPOSE 8000
WORKDIR ./

COPY . /
RUN pip install -e .