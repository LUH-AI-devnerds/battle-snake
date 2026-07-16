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
# Patch hisss at image build time (site-packages is writable here; runtime-only patch
# can fail silently on read-only Railway layers and bring back FALLBACK_MOVE=up).
RUN python -c "from battlesnake_ai.env.hisss_view_radius_fix import apply_view_radius_row_index_fix; import sys; sys.exit(0 if apply_view_radius_row_index_fix() else 1)"

ENV BATTLE_SNAKE_CHECKPOINT="best_checkpoint/rainbow_20260704_125842_ep1600.pt"
ENV SNAKE_AUTHOR="the sea snake"
ENV SNAKE_COLOR="#4488ff"
# Survival/combat layer: avoid equal/longer heads; grow then hunt shorter snakes.
ENV SURVIVAL_FILTER="1"
ENV SURVIVAL_HUNGER_HEALTH="35"
ENV SURVIVAL_STRATEGY="aggressive"

EXPOSE 8000
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
