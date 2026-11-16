"""
Tests for the Celery pipeline task and enqueue_pipeline_job().

All tests run without Redis or a real Celery worker using Celery's
ALWAYS_EAGER mode, which runs tasks synchronously in the same process.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ── enqueue_pipeline_job tests ────────────────────────────────────────────────

class TestEnqueuePipelineJob:
    def test_uses_thread_fallback_when_celery_disabled(self):
        """USE_CELERY=false → runs in a background thread, no Redis needed."""
        from app.services.pipeline_service import enqueue_pipeline_job

        with patch("app.services.pipeline_service.settings") as mock_settings, \
             patch("app.services.pipeline_service.run_pipeline_job") as mock_job, \
             patch("threading.Thread") as mock_thread:
            mock_settings.use_celery = False
            thread_instance = MagicMock()
            mock_thread.return_value = thread_instance

            task_id = enqueue_pipeline_job("session-123")

        assert "session-123" in task_id
        mock_thread.assert_called_once()
        thread_instance.start.assert_called_once()

    def test_uses_celery_when_enabled(self):
        """USE_CELERY=true → calls run_pipeline_task.delay()."""
        from app.services.pipeline_service import enqueue_pipeline_job

        mock_async_result = MagicMock()
        mock_async_result.id = "celery-task-abc123"

        with patch("app.services.pipeline_service.settings") as mock_settings, \
             patch("app.tasks.run_pipeline_task") as mock_task:
            mock_settings.use_celery = True
            mock_task.delay.return_value = mock_async_result

            task_id = enqueue_pipeline_job("session-456")

        assert task_id == "celery-task-abc123"
        mock_task.delay.assert_called_once_with("session-456")

    def test_returns_string_task_id_in_both_modes(self):
        """Task ID is always a string regardless of mode."""
        from app.services.pipeline_service import enqueue_pipeline_job

        with patch("app.services.pipeline_service.settings") as mock_settings, \
             patch("threading.Thread"):
            mock_settings.use_celery = False
            result = enqueue_pipeline_job("session-789")

        assert isinstance(result, str)


# ── run_pipeline_job re-raise test ────────────────────────────────────────────

class TestPipelineJobReRaise:
    def test_reraises_exception_so_celery_can_retry(self):
        """
        run_pipeline_job now re-raises exceptions after marking the session
        failed — this is what lets Celery's retry logic kick in.
        """
        from app.services.pipeline_service import run_pipeline_job

        mock_session = MagicMock()
        mock_session.status = "uploaded"
        mock_session.session_type = "professional"

        mock_db = MagicMock()
        mock_db.get.return_value = mock_session

        with patch("app.services.pipeline_service.SessionLocal", return_value=mock_db), \
             patch("app.services.pipeline_service.settings") as mock_settings, \
             patch("app.services.pipeline_service.broadcast_sync"), \
             patch("app.services.pipeline_service.generate_mock_result",
                   side_effect=RuntimeError("OpenAI timeout")):
            mock_settings.use_mock_pipeline = True

            with pytest.raises(RuntimeError, match="OpenAI timeout"):
                run_pipeline_job("session-fail")

        # session should be marked failed before the re-raise
        assert mock_session.status == "failed"
        assert "OpenAI timeout" in mock_session.error_message


# ── Celery task tests (eager mode) ────────────────────────────────────────────

class TestCeleryTask:
    def test_task_calls_run_pipeline_job(self):
        """run_pipeline_task calls run_pipeline_job with the session_id."""
        from app.tasks import run_pipeline_task

        with patch("app.tasks.run_pipeline_job") as mock_job:
            # apply() runs the task synchronously (no worker needed)
            result = run_pipeline_task.apply(args=["session-celery-1"])

        mock_job.assert_called_once_with("session-celery-1")
        assert result.result["session_id"] == "session-celery-1"
        assert result.result["status"] == "completed"

    def test_task_is_registered_on_celery_app(self):
        """The task name is registered so workers can find it."""
        from app.celery_app import celery
        assert "app.tasks.run_pipeline_task" in celery.tasks

    def test_task_has_retry_config(self):
        """Task is configured with max_retries=3."""
        from app.tasks import run_pipeline_task
        assert run_pipeline_task.max_retries == 3
