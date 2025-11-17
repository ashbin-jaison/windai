# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Prevent Python from writing .pyc and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install minimal OS deps (libgomp for xgboost/lightgbm)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project
COPY . /app

# Railway provides PORT env var; default to 8000 for local runs
ENV PORT=8000

# Start via gunicorn (threads for lightweight concurrency)
# Module path points to package `app` and module `app.py` exporting `app`
CMD exec gunicorn -w 2 -k gthread --threads 8 -b 0.0.0.0:${PORT} app.app:app
