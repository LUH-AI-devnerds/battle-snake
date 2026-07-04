FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt agent/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py ./
COPY agent/src/battlesnake_ai ./agent/src/battlesnake_ai
COPY logs/checkpoints/ ./logs/checkpoints/

ENV PYTHONPATH=/app/agent/src
ENV BATTLE_SNAKE_CHECKPOINT=logs/checkpoints/rainbow_20260602_182838_ep75.pt
ENV SNAKE_AUTHOR=Battle Snake
ENV SNAKE_COLOR=#4488ff

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
