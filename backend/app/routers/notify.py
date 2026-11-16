"""
POST /sessions/{session_id}/notify

Sends action item emails to meeting participants after a session completes.

The caller supplies a mapping of speaker names → email addresses. MeetMind
doesn't store participant emails — the host knows who was in the meeting
and provides this mapping at notification time.

Request body:
  {
    "owner_emails": {
      "Alice": "alice@company.com",
      "Bob":   "bob@company.com"
    }
  }

Response:
  {
    "sent": 2,
    "skipped": 0,
    "failed": 0,
    "results": [
      {"owner": "Alice", "email": "alice@...", "sent": true},
      {"owner": "Bob",   "email": "bob@...",   "sent": true}
    ]
  }

Only professional sessions have action items. Calling this on a student
session returns 400.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session as DBSession

from ..auth import get_current_user
from ..database import get_db
from ..models import MeetingSession, User
from ..services.email_service import send_action_item_emails

router = APIRouter(prefix="/sessions", tags=["notifications"])


class NotifyRequest(BaseModel):
    owner_emails: dict[str, EmailStr]   # {"Alice": "alice@example.com"}


class NotifyResult(BaseModel):
    owner: str
    email: str
    sent: bool
    error: str | None = None


class NotifyResponse(BaseModel):
    sent: int
    skipped: int
    failed: int
    results: list[NotifyResult]


@router.post("/{session_id}/notify", response_model=NotifyResponse)
def notify_participants(
    session_id: str,
    body: NotifyRequest,
    db: DBSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Send action item emails to meeting participants.

    Only the session owner can trigger notifications.
    Only completed professional sessions have action items.
    """
    session = db.get(MeetingSession, session_id)
    if session is None or session.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found.")

    if session.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Session is not completed — current status: {session.status}.",
        )

    if session.session_type != "professional":
        raise HTTPException(
            status_code=400,
            detail="Action item emails are only available for professional sessions.",
        )

    output = session.professional_output_json or {}
    action_items = output.get("action_items", [])

    if not action_items:
        raise HTTPException(
            status_code=400,
            detail="No action items found in this session.",
        )

    if not body.owner_emails:
        raise HTTPException(
            status_code=422,
            detail="owner_emails cannot be empty.",
        )

    # format the session date for the email body
    completed = session.completed_at or session.created_at
    session_date = completed.strftime("%B %d, %Y") if completed else "Unknown date"

    results = send_action_item_emails(
        session_id=session_id,
        session_title=session.title,
        session_date=session_date,
        action_items_json=action_items,
        owner_emails=dict(body.owner_emails),
    )

    sent = sum(1 for r in results if r.sent)
    failed = sum(1 for r in results if not r.sent)

    # owners with action items but no email provided
    owners_with_items = {item.get("owner") for item in action_items}
    skipped = len(owners_with_items - set(body.owner_emails.keys()))

    return NotifyResponse(
        sent=sent,
        skipped=skipped,
        failed=failed,
        results=[
            NotifyResult(
                owner=r.owner,
                email=r.email,
                sent=r.sent,
                error=r.error,
            )
            for r in results
        ],
    )
