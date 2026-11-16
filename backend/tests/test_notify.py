"""
Tests for action item email notifications.

All tests run without real email sending — Resend is mocked.
Tests cover: email grouping by owner, HTML/plain text content,
endpoint auth, 400/404 error cases, and dev-mode logging.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ── email_service unit tests ──────────────────────────────────────────────────

class TestEmailService:
    _ACTION_ITEMS = [
        {"owner": "Alice", "task": "Send the proposal", "due": "Friday"},
        {"owner": "Alice", "task": "Review the budget", "due": None},
        {"owner": "Bob",   "task": "Update the roadmap", "due": "Nov 20"},
    ]

    def test_sends_one_email_per_owner(self):
        from app.services.email_service import send_action_item_emails
        with patch("app.services.email_service._send_via_resend") as mock_send, \
             patch.dict("os.environ", {"RESEND_API_KEY": "re_fake_key"}):
            results = send_action_item_emails(
                session_id="s1",
                session_title="Weekly Standup",
                session_date="November 15, 2026",
                action_items_json=self._ACTION_ITEMS,
                owner_emails={"Alice": "alice@test.com", "Bob": "bob@test.com"},
            )
        assert mock_send.call_count == 2
        owners = {r.owner for r in results}
        assert owners == {"Alice", "Bob"}

    def test_skips_owner_with_no_email(self):
        from app.services.email_service import send_action_item_emails
        with patch("app.services.email_service._send_via_resend") as mock_send, \
             patch.dict("os.environ", {"RESEND_API_KEY": "re_fake_key"}):
            results = send_action_item_emails(
                session_id="s1",
                session_title="Weekly Standup",
                session_date="November 15, 2026",
                action_items_json=self._ACTION_ITEMS,
                owner_emails={"Alice": "alice@test.com"},  # Bob has no email
            )
        assert mock_send.call_count == 1
        assert len(results) == 1
        assert results[0].owner == "Alice"

    def test_alice_email_contains_only_her_tasks(self):
        from app.services.email_service import send_action_item_emails
        captured = []
        def capture_send(to, subject, html, plain, from_addr, api_key):
            captured.append({"to": to, "plain": plain})

        with patch("app.services.email_service._send_via_resend", side_effect=capture_send), \
             patch.dict("os.environ", {"RESEND_API_KEY": "re_fake_key"}):
            send_action_item_emails(
                session_id="s1",
                session_title="Standup",
                session_date="Nov 15",
                action_items_json=self._ACTION_ITEMS,
                owner_emails={"Alice": "alice@test.com", "Bob": "bob@test.com"},
            )

        alice_email = next(c for c in captured if c["to"] == "alice@test.com")
        assert "Send the proposal" in alice_email["plain"]
        assert "Review the budget" in alice_email["plain"]
        assert "Update the roadmap" not in alice_email["plain"]  # Bob's task

    def test_due_date_shown_in_email(self):
        from app.services.email_service import _build_plain
        plain = _build_plain("Alice", "Standup", [
            {"task": "Send proposal", "due": "Friday"},
            {"task": "Review budget", "due": None},
        ])
        assert "due: Friday" in plain
        assert "Review budget" in plain

    def test_html_contains_owner_name(self):
        from app.services.email_service import _build_html
        html = _build_html("Alice", "Standup", "Nov 15", [
            {"task": "Send proposal", "due": "Friday"}
        ])
        assert "Alice" in html
        assert "Send proposal" in html
        assert "Friday" in html

    def test_dev_mode_logs_instead_of_sending(self):
        """No RESEND_API_KEY → dev mode logs but reports sent=True."""
        from app.services.email_service import send_action_item_emails
        with patch.dict("os.environ", {}, clear=True):  # no RESEND_API_KEY
            with patch("app.services.email_service._send_via_resend") as mock_send:
                results = send_action_item_emails(
                    session_id="s1",
                    session_title="Standup",
                    session_date="Nov 15",
                    action_items_json=[{"owner": "Alice", "task": "Send report", "due": None}],
                    owner_emails={"Alice": "alice@test.com"},
                )
        mock_send.assert_not_called()   # Resend NOT called in dev mode
        assert results[0].sent is True  # still reports success for dev convenience

    def test_failed_send_recorded_in_result(self):
        from app.services.email_service import send_action_item_emails
        with patch("app.services.email_service._send_via_resend",
                   side_effect=Exception("SMTP timeout")), \
             patch.dict("os.environ", {"RESEND_API_KEY": "re_fake_key"}):
            results = send_action_item_emails(
                session_id="s1",
                session_title="Standup",
                session_date="Nov 15",
                action_items_json=[{"owner": "Alice", "task": "Send report", "due": None}],
                owner_emails={"Alice": "alice@test.com"},
            )
        assert results[0].sent is False
        assert "SMTP timeout" in results[0].error


# ── notify endpoint tests ─────────────────────────────────────────────────────


# ── notify endpoint tests ─────────────────────────────────────────────────────

from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.routers import notify


# Create test application
test_app = FastAPI()
test_app.include_router(notify.router)


def _mock_session(
    status="completed",
    session_type="professional",
    action_items=None,
):
    s = MagicMock()
    s.id = "sess-1"
    s.owner_id = "user-1"
    s.title = "Weekly Standup"
    s.status = status
    s.session_type = session_type
    s.completed_at = None
    s.created_at = None

    # Important: preserve an explicitly empty list
    if action_items is None:
        action_items = [
            {"owner": "Alice", "task": "Send proposal", "due": "Friday"},
            {"owner": "Bob", "task": "Update roadmap", "due": None},
        ]

    s.professional_output_json = {
        "action_items": action_items
    }
    return s


def _mock_user():
    u = MagicMock()
    u.id = "user-1"
    return u


@pytest.fixture
def client():
    def override_get_current_user():
        return _mock_user()

    def override_get_db():
        yield MagicMock()

    test_app.dependency_overrides[notify.get_current_user] = (
        override_get_current_user
    )
    test_app.dependency_overrides[notify.get_db] = override_get_db

    try:
        with TestClient(
            test_app,
            raise_server_exceptions=False
        ) as test_client:
            yield test_client
    finally:
        test_app.dependency_overrides.clear()


class TestNotifyEndpoint:

    def _patch(self, session=None):
        db = MagicMock()
        db.get.return_value = (
            session if session is not None else _mock_session()
        )

        def override_get_db():
            yield db

        test_app.dependency_overrides[notify.get_db] = override_get_db

        return db

    def test_returns_404_for_unknown_session(self, client):
        db = MagicMock()
        db.get.return_value = None

        def override_get_db():
            yield db

        test_app.dependency_overrides[notify.get_db] = override_get_db

        res = client.post(
            "/sessions/unknown/notify",
            json={"owner_emails": {"Alice": "a@test.com"}}
        )

        assert res.status_code == 404

    def test_returns_400_for_non_completed_session(self, client):
        self._patch(
            session=_mock_session(status="processing")
        )

        res = client.post(
            "/sessions/sess-1/notify",
            json={"owner_emails": {"Alice": "a@test.com"}}
        )

        assert res.status_code == 400
        assert "not completed" in res.json()["detail"]

    def test_returns_400_for_student_session(self, client):
        self._patch(
            session=_mock_session(session_type="student")
        )

        res = client.post(
            "/sessions/sess-1/notify",
            json={"owner_emails": {"Alice": "a@test.com"}}
        )

        assert res.status_code == 400
        assert "professional" in res.json()["detail"]

    def test_returns_400_when_no_action_items(self, client):
        self._patch(
            session=_mock_session(action_items=[])
        )

        res = client.post(
            "/sessions/sess-1/notify",
            json={"owner_emails": {"Alice": "a@test.com"}}
        )

        assert res.status_code == 400
        assert "No action items" in res.json()["detail"]

    def test_successful_notify_returns_sent_counts(self, client):
        self._patch()

        with patch(
            "app.services.email_service._send_via_resend"
        ), patch.dict(
            "os.environ",
            {"RESEND_API_KEY": "re_fake"}
        ):
            res = client.post(
                "/sessions/sess-1/notify",
                json={
                    "owner_emails": {
                        "Alice": "alice@test.com",
                        "Bob": "bob@test.com",
                    }
                },
            )

        assert res.status_code == 200

        data = res.json()
        assert data["sent"] == 2
        assert data["failed"] == 0
        assert len(data["results"]) == 2

    def test_skipped_count_for_owner_without_email(self, client):
        self._patch()

        with patch(
            "app.services.email_service._send_via_resend"
        ), patch.dict(
            "os.environ",
            {"RESEND_API_KEY": "re_fake"}
        ):
            res = client.post(
                "/sessions/sess-1/notify",
                json={
                    "owner_emails": {
                        "Alice": "alice@test.com"
                    }
                },
            )

        data = res.json()
        assert res.status_code == 200
        assert data["sent"] == 1
        assert data["skipped"] == 1
