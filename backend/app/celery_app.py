"""
Celery application instance for MeetMind's background pipeline.

Why Celery instead of FastAPI BackgroundTasks:
  BackgroundTasks runs in the same process as the web server. If Whisper takes
  3 minutes on a 1-hour recording and two users upload at the same time, they
  block each other — the second user's pipeline waits for the first to finish.
  Celery runs workers in separate processes, so N workers handle N sessions
  concurrently. It also adds:
    - Automatic retries on transient failures (OpenAI timeout, Pinecone blip)
    - Task status tracking (PENDING → STARTED → SUCCESS/FAILURE)
    - Persistent task queue — if the server restarts mid-pipeline, the task
      re-queues instead of silently disappearing

Broker and backend: Redis.
  Redis is already in requirements.txt and used for pubsub/room_manager.
  CELERY_BROKER_URL defaults to redis://localhost:6379/0.
  CELERY_RESULT_BACKEND defaults to redis://localhost:6379/1 (separate DB).

Usage:
  Start a worker:
    celery -A app.celery_app worker --loglevel=info --concurrency=2

  The web process (uvicorn) enqueues tasks; workers pick them up independently.
  Both read the same .env — no extra config needed.
"""
import os

from celery import Celery
from dotenv import load_dotenv

load_dotenv()

_BROKER = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

celery = Celery(
    "meetmind",
    broker=_BROKER,
    backend=_BACKEND,
    include=["app.tasks"],   # tells Celery where to find the task definitions
)

celery.conf.update(
    # serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # retry policy — retries 3 times with exponential back-off (2s, 4s, 8s)
    task_acks_late=True,            # only ack after the task finishes, not on receipt
    task_reject_on_worker_lost=True,# re-queue if the worker crashes mid-task

    # result expiry — keep task results for 24 hours then auto-delete from Redis
    result_expires=86_400,

    # routing — all pipeline tasks go to the "pipeline" queue
    # start a dedicated worker with: celery -A app.celery_app worker -Q pipeline
    task_routes={"app.tasks.run_pipeline_task": {"queue": "pipeline"}},
)
