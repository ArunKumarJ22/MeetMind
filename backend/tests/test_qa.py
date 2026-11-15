
"""
Tests for POST /sessions/{session_id}/ask (Q&A streaming endpoint).

All tests use FastAPI's TestClient.
No real OpenAI calls or database setup are required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.qa import router, _build_context


# ── Minimal test application ──────────────────────────────────────────────────

app_under_test = FastAPI()
app_under_test.include_router(router)


# ── Mock helpers ──────────────────────────────────────────────────────────────

def _mock_session(
    session_id="sess-1",
    status="completed",
    session_type="professional",
    transcript=None,
    professional_output=None,
):
    session_obj = MagicMock()
    session_obj.id = session_id
    session_obj.owner_id = "user-1"
    session_obj.title = "Weekly Standup"
    session_obj.status = status
    session_obj.session_type = session_type

    session_obj.transcript_json = transcript or {
        "segments": [
            {
                "start": 0.0,
                "speaker": "Host",
                "text": "Let's start the standup.",
            },
            {
                "start": 5.0,
                "speaker": "Alice",
                "text": "I finished the backend integration.",
            },
            {
                "start": 10.0,
                "speaker": "Bob",
                "text": "I'll send the report by Friday.",
            },
        ]
    }

    session_obj.professional_output_json = professional_output or {
        "role_summaries": [
            {
                "role": "Host",
                "summary": "Ran the standup. Requested report from Bob.",
            },
            {
                "role": "Alice",
                "summary": "Finished backend integration.",
            },
            {
                "role": "Bob",
                "summary": "Will send the report by Friday.",
            },
        ]
    }

    session_obj.student_output_json = None

    return session_obj


def _mock_current_user():
    user = MagicMock()
    user.id = "user-1"
    return user


# ── Client fixture ────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    app_under_test.dependency_overrides.clear()

    with TestClient(app_under_test) as test_client:
        yield test_client

    app_under_test.dependency_overrides.clear()


# ── Dependency override helper ───────────────────────────────────────────────


def _patch_deps(session_obj=None, user=None, db_mock=None):
    from app.routers import qa

    if db_mock is None:
        db_mock = MagicMock()
        db_mock.get.return_value = (
            session_obj if session_obj is not None else _mock_session()
        )

    current_user = user if user is not None else _mock_current_user()

    # Return the database mock directly, NOT iter([db_mock]).
    app_under_test.dependency_overrides[qa.get_db] = lambda: db_mock

    app_under_test.dependency_overrides[qa.get_current_user] = (
        lambda: current_user
    )



# ── Context-building tests ───────────────────────────────────────────────────

class TestBuildContext:

    def test_includes_speaker_names(self):
        session_obj = _mock_session()
        context = _build_context(session_obj)

        assert "Alice" in context
        assert "Bob" in context

    def test_includes_transcript_text(self):
        session_obj = _mock_session()
        context = _build_context(session_obj)

        assert "backend integration" in context

    def test_includes_role_summaries(self):
        session_obj = _mock_session()
        context = _build_context(session_obj)

        assert "ROLE SUMMARIES" in context
        assert "Will send the report by Friday" in context

    def test_truncates_very_long_transcripts(self):
        long_text = "word " * 5000

        session_obj = _mock_session(
            transcript={
                "segments": [
                    {
                        "start": 0,
                        "speaker": "Speaker",
                        "text": long_text,
                    }
                ]
            }
        )

        context = _build_context(session_obj)

        assert len(context) <= 12_500
        assert "[transcript truncated]" in context

    def test_handles_missing_transcript_gracefully(self):
        session_obj = _mock_session(transcript={})
        context = _build_context(session_obj)

        assert "ROLE SUMMARIES" in context

    def test_handles_no_speaker_on_segment(self):
        session_obj = _mock_session(
            transcript={
                "segments": [
                    {
                        "start": 0,
                        "text": "No speaker label here.",
                    }
                ]
            }
        )

        context = _build_context(session_obj)

        assert "No speaker label here" in context
        assert "Speaker:" in context


# ── Endpoint tests ───────────────────────────────────────────────────────────

class TestAskEndpoint:

    def test_returns_404_for_unknown_session(self, client):
        db_mock = MagicMock()
        db_mock.get.return_value = None

        _patch_deps(db_mock=db_mock)

        response = client.post(
            "/sessions/unknown/ask",
            json={"question": "What happened?"},
        )

        assert response.status_code == 404

    def test_returns_400_for_non_completed_session(self, client):
        _patch_deps(
            session_obj=_mock_session(status="processing")
        )

        response = client.post(
            "/sessions/sess-1/ask",
            json={"question": "What happened?"},
        )

        assert response.status_code == 400
        assert "not ready" in response.json()["detail"].lower()

    def test_returns_404_if_session_owned_by_different_user(self, client):
        session_obj = _mock_session()
        session_obj.owner_id = "different-user"

        _patch_deps(session_obj=session_obj)

        response = client.post(
            "/sessions/sess-1/ask",
            json={"question": "What happened?"},
        )

        assert response.status_code == 404
