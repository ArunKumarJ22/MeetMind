"""
POST /sessions/{session_id}/ask  —  Q&A chat over a completed MeetMind session.

What this does:
  1. Loads the session's transcript from the DB (already computed by the pipeline)
  2. Builds a context from the transcript's raw_text + role summaries
  3. Asks GPT to answer the question using ONLY that context (no hallucination)
  4. Streams the answer token-by-token back to the frontend via Server-Sent Events

Why streaming:
  A GPT answer can take 10-20 seconds on a long transcript. Streaming means the
  user sees tokens arriving immediately instead of staring at a spinner. This is
  one of the most noticeable UX improvements and is a common interview question.

Design decision — why NOT use Pinecone here:
  retrieve_for_mention() in rag.py is designed for LIVE meeting retrieval against
  pre-uploaded reference documents. For post-meeting Q&A, the transcript is already
  in the DB and is short enough to fit in GPT's context window. Using it directly
  avoids a Pinecone round-trip and works even when no documents were uploaded.
  Future improvement: for very long transcripts (>2 hours), switch to chunked
  vector retrieval.
"""
from __future__ import annotations

import json
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel
from sqlalchemy.orm import Session as DBSession

from ..auth import get_current_user
from ..config import settings
from ..database import get_db
from ..models import MeetingSession, User

router = APIRouter(prefix="/sessions", tags=["qa"])

_MAX_CONTEXT_CHARS = 12_000  # ~3k tokens — well within gpt-4o-mini's window


class AskRequest(BaseModel):
    question: str


def _build_context(session: MeetingSession) -> str:
    """
    Builds the Q&A context from what the pipeline already computed:
      - transcript segments (speaker + text, with timestamps)
      - role summaries (host summary + per-participant summaries)

    We prefer the transcript over the summaries alone because the transcript
    contains the exact words said, which matters for faithfulness.
    """
    parts: list[str] = []

    # ── transcript segments ───────────────────────────────────────────────────
    transcript = session.transcript_json or {}
    segments = transcript.get("segments", [])
    if segments:
        lines = []
        for seg in segments:
            speaker = seg.get("speaker") or "Speaker"
            text = seg.get("text", "").strip()
            start = seg.get("start", 0)
            if text:
                lines.append(f"[{start:.0f}s] {speaker}: {text}")
        if lines:
            parts.append("MEETING TRANSCRIPT:\n" + "\n".join(lines))

    # ── role summaries as additional context ──────────────────────────────────
    # pick professional or student output depending on session type
    output = (
        session.professional_output_json
        if session.session_type == "professional"
        else session.student_output_json
    ) or {}

    role_summaries = output.get("role_summaries", [])
    if role_summaries:
        summary_lines = [f"{s['role']}: {s['summary']}" for s in role_summaries]
        parts.append("ROLE SUMMARIES:\n" + "\n".join(summary_lines))

    context = "\n\n".join(parts)

    # truncate to avoid exceeding context window on very long meetings
    if len(context) > _MAX_CONTEXT_CHARS:
        context = context[:_MAX_CONTEXT_CHARS] + "\n\n[transcript truncated]"

    return context


async def _stream_answer(
    question: str,
    context: str,
    session_title: str,
) -> AsyncGenerator[str, None]:
    """
    Calls GPT with streaming and yields Server-Sent Event lines.
    The frontend reads these with an EventSource or fetch + ReadableStream.

    SSE format:
      data: {"type": "token", "content": "Hello"}
      data: {"type": "done"}
      data: {"type": "error", "content": "..."}
    """
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    system_prompt = (
        f"You are a helpful assistant answering questions about the meeting titled "
        f'"{session_title}".\n\n'
        "Answer using ONLY the information in the transcript and summaries provided. "
        "If the answer is not in the context, say so clearly — do not guess or use "
        "outside knowledge.\n\n"
        "When relevant, reference the speaker by name and the approximate timestamp. "
        "Be concise but complete.\n\n"
        f"CONTEXT:\n{context}"
    )

    try:
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            stream=True,
            temperature=0.0,
            max_tokens=500,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                payload = json.dumps({"type": "token", "content": delta.content})
                yield f"data: {payload}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as exc:
        payload = json.dumps({"type": "error", "content": str(exc)})
        yield f"data: {payload}\n\n"


@router.post("/{session_id}/ask")
async def ask_session(
    session_id: str,
    body: AskRequest,
    db: DBSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Stream a GPT answer to a question about a completed session's transcript.

    Returns a text/event-stream (Server-Sent Events) response.
    The frontend reads tokens as they arrive using EventSource or fetch streaming.

    Errors:
      404 — session not found or not owned by the current user
      400 — session is not yet completed (transcript not available)
      422 — question is empty
    """
    if not body.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    session = db.get(MeetingSession, session_id)
    if session is None or session.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found.")

    if session.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Session is not ready for Q&A — current status: {session.status}.",
        )

    if not session.transcript_json:
        raise HTTPException(
            status_code=400,
            detail="Transcript not available for this session.",
        )

    context = _build_context(session)

    return StreamingResponse(
        _stream_answer(body.question, context, session.title),
        media_type="text/event-stream",
        headers={
            # prevent nginx/proxy from buffering the SSE stream
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
        },
    )
