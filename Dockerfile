FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt ./requirements-api.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-api.txt

COPY server.py ./server.py
COPY app ./app
COPY web ./web
COPY models ./models

EXPOSE 8080

# One worker is intentional: server.py keeps one shared model in memory.
CMD ["sh", "-c", "exec uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
