FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-server.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-server.txt

COPY server.py ./
COPY agent/src/battlesnake_ai ./agent/src/battlesnake_ai
COPY best_checkpoint/ ./best_checkpoint/

ENV PYTHONPATH=/app/agent/src
ENV BATTLE_SNAKE_CHECKPOINT="best_checkpoint/rainbow_20260704_125842_ep1600.pt"
ENV SNAKE_AUTHOR="the sea snake"
ENV SNAKE_COLOR="#4488ff"

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
