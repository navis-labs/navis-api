FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# No build-essential / libpq-dev here, deliberately. asyncpg, bcrypt,
# cryptography and greenlet all publish manylinux wheels for cp312, and nothing
# in this project uses psycopg — so the toolchain compiled nothing and shipped
# in the final image anyway. Removing it took the image from 1.04GB to 605MB,
# which is 435MB off every ECR push, every task cold start and every rollback.
#
# If a future dependency genuinely needs to compile, add a builder stage rather
# than putting the toolchain back in the runtime image.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Run as a normal user. Nothing here needs root, and a container that does is a
# container where a code-execution bug owns the whole filesystem.
RUN useradd --create-home --uid 10001 navis && chown -R navis:navis /app
USER navis

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
