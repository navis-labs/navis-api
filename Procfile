# Non-Docker (Heroku / Railway / Render) process model.
#
# ALL THREE are required. Running `web` alone ships an API in front of a pipeline
# that never advances: signals are written, outbox pointers are staged, and
# nothing relays them into memory.
#
# See docker-compose.yml for the same topology under Docker, and for why the
# worker must use --pool=prefork (never threads: run_async holds one event loop
# per process and asyncpg pools bind to it).
#
# Scale `worker` freely; keep `beat` at EXACTLY ONE dyno.
web: uvicorn navis.main:app --host 0.0.0.0 --port ${PORT:-8000}
worker: celery -A navis.platform.queue.celery_app:celery_app worker --pool=prefork --concurrency=2 --loglevel=INFO
beat: celery -A navis.platform.queue.celery_app:celery_app beat --loglevel=INFO
release: alembic upgrade head
