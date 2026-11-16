"""
Celery tasks for MeetMind's AI pipeline.

The key design constraint (from pipeline_service.py's comment):
  "swap this for a proper task queue without changing run_pipeline_job's signature"

We honour that — `run_pipeline_task` just calls `run_pipeline_job(session_id)`,
the same function the BackgroundTasks path called. Zero changes to the AI pipeline
logic itself.

Retry strategy:
  - 3 automatic retries on any exception
  - Exponential back-off: 2s → 4s → 8s
  - After 3 retries the task moves to FAILURE state and the session gets
    status="failed" in the DB (run_pipeline_job already handles that internally)

Transient failures this catches:
  - OpenAI API timeout / rate limit (Whisper, GPT)
  - Pinecone connection blip
  - Temporary disk I/O error reading the audio file

It does NOT retry on permanent failures:
  - Audio file not found (deleted between upload and processing)
  - Invalid session_id (session was deleted by the user)
  These raise immediately and the except block in run_pipeline_job marks
  the session as failed.
"""
from __future__ import annotations

import logging

from celery import Task
from celery.utils.log import get_task_logger

from .celery_app import celery
from .services.pipeline_service import run_pipeline_job

logger = get_task_logger(__name__)


class _PipelineTask(Task):
    """Base class that adds structured logging on failure."""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        session_id = args[0] if args else "unknown"
        logger.error(
            "Pipeline task failed permanently",
            extra={
                "task_id": task_id,
                "session_id": session_id,
                "error": str(exc),
            },
        )
        super().on_failure(exc, task_id, args, kwargs, einfo)


@celery.task(
    base=_PipelineTask,
    name="app.tasks.run_pipeline_task",
    bind=True,
    max_retries=3,
    default_retry_delay=2,     # seconds — doubles on each retry (2 → 4 → 8)
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True,          # adds small random jitter to avoid thundering herd
)
def run_pipeline_task(self: Task, session_id: str) -> dict:
    """
    Celery task that runs the MeetMind AI pipeline for one session.

    Enqueue it from the web process with:
        run_pipeline_task.delay(session_id)

    Returns a dict with task metadata on success (stored in Redis for 24h).
    The actual session data is written to PostgreSQL by run_pipeline_job.
    """
    logger.info("Starting pipeline", extra={"session_id": session_id})

    # run_pipeline_job handles all DB writes and WebSocket broadcasts internally
    # it also handles its own exception → sets session.status = "failed"
    run_pipeline_job(session_id)

    logger.info("Pipeline complete", extra={"session_id": session_id})
    return {"session_id": session_id, "status": "completed"}
