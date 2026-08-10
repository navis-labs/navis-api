FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for asyncpg / cryptography builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8000

# This image serves THREE roles; the default command is only one of them.
#
#   api     CMD as below
#   worker  celery -A navis.platform.queue.celery_app:celery_app worker --pool=prefork
#   beat    celery -A navis.platform.queue.celery_app:celery_app beat
#
# Running the API alone ships a product where Gmail never becomes memory: signals
# and their outbox pointers are written and nothing ever relays them. Deploy all
# three with a restart policy — see docker-compose.yml / Procfile.
CMD ["uvicorn", "navis.main:app", "--host", "0.0.0.0", "--port", "8000"]
