"""
Bridges the web layer to the ai-pipeline package.

v2: Celery replaces FastAPI BackgroundTasks.

Before (BackgroundTasks):
  background_tasks.add_task(run_pipeline_job, session.id)
  → ran in the same uvicorn process
  → concurrent uploads blocked each other (Whisper is CPU-bound)
  → if the server restarted mid-pipeline, the task silently disappeared

After (Celery):
  enqueue_pipeline_job(session.id)
  → enqueues to Redis; a Celery worker picks it up independently
  → N workers = N concurrent sessions
  → task survives server restarts (re-queued from Redis)
  → 3 automatic retries on transient failures (OpenAI timeout, Pinecone blip)

run_pipeline_job() is unchanged — it still does all the DB writes and
WebSocket broadcasts. The callers (sessions.py router) now call
enqueue_pipeline_job() instead of background_tasks.add_task(run_pipeline_job).
"""
from datetime import datetime, timezone

from meetmind_ai import generate_mock_result, run_pipeline
from meetmind_ai.schemas import SessionType

from ..config import settings
from ..database import SessionLocal
from ..models import MeetingSession
from .storage_service import get_local_path
from .ws_manager import broadcast_sync


def run_pipeline_job(session_id: str) -> None:
    """
    Runs the full AI pipeline for one session and writes results to the DB.
    Called by the Celery task (tasks.py). Signature is unchanged from v1
    so the task wrapper is a one-liner.
    """
    db = SessionLocal()
    try:
        session = db.get(MeetingSession, session_id)
        if session is None:
            return

        session.status = "processing"
        db.commit()
        broadcast_sync(session_id, {"status": "processing"})

        try:
            session_type = SessionType(session.session_type)
            if settings.use_mock_pipeline:
                result = generate_mock_result(session_id, session_type)
            else:
                local_path = get_local_path(session.audio_path)
                on_stage = lambda stage: broadcast_sync(
                    session_id, {"status": "processing", "stage": stage}
                )
                result = run_pipeline(local_path, session_id, session_type, on_stage=on_stage)

            session.status = "completed"
            session.completed_at = datetime.now(timezone.utc)
            session.processing_seconds = result.processing_seconds
            session.transcript_json = result.transcript.model_dump()
            session.student_output_json = (
                result.student_output.model_dump() if result.student_output else None
            )
            session.professional_output_json = (
                result.professional_output.model_dump() if result.professional_output else None
            )
            session.retrieved_documents_json = [d.model_dump() for d in result.retrieved_documents]
            db.commit()
            broadcast_sync(session_id, {"status": "completed"})

        except Exception as exc:  # noqa: BLE001
            session.status = "failed"
            session.error_message = str(exc)
            db.commit()
            broadcast_sync(session_id, {"status": "failed", "error": str(exc)})
            raise   # re-raise so Celery's retry logic can catch it

    finally:
        db.close()


def enqueue_pipeline_job(session_id: str) -> str:
    """
    Enqueues the pipeline job to Celery and returns the Celery task ID.

    Falls back to running synchronously in a thread if Celery/Redis is not
    available (USE_CELERY=false in .env) — keeps local dev working without
    needing Redis running.

    Returns the Celery task ID (useful for debugging, not exposed to the client).
    """
    if not settings.use_celery:
        # local dev fallback — run in a background thread like the old code
        import threading
        t = threading.Thread(target=run_pipeline_job, args=(session_id,), daemon=True)
        t.start()
        return f"thread-{session_id}"

    from ..tasks import run_pipeline_task
    result = run_pipeline_task.delay(session_id)
    return result.id
