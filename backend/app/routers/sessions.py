from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session as DBSession

from ..auth import get_current_user
from ..config import settings
from ..database import get_db
from ..rate_limit import limiter
from ..models import MeetingSession, Room, User
from ..schemas import SessionListItem, SessionOut
from ..services.pipeline_service import run_pipeline_job
from ..services.rooms import generate_room_code
from ..services.storage_service import delete_audio, save_audio

router = APIRouter(prefix="/sessions", tags=["sessions"])

_VALID_SESSION_TYPES = {"student", "professional"}


class LiveSessionIn(BaseModel):
    title: str
    session_type: str


def _check_upload_size(file_bytes: bytes) -> None:
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file exceeds the {settings.max_upload_size_mb}MB limit.",
        )


@router.post("", response_model=SessionOut, status_code=201)
@limiter.limit("10/minute")
async def create_session(
    request: Request,
    background_tasks: BackgroundTasks,
    title: str = Form(...),
    session_type: str = Form(...),
    audio: UploadFile = File(...),
    db: DBSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if session_type not in _VALID_SESSION_TYPES:
        raise HTTPException(status_code=400, detail=f"session_type must be one of {_VALID_SESSION_TYPES}")

    file_bytes = await audio.read()
    _check_upload_size(file_bytes)

    session = MeetingSession(
        owner_id=current_user.id, title=title, session_type=session_type,
        status="uploaded", audio_path="",
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    session.audio_path = save_audio(file_bytes, audio.filename or "audio.webm", session.id)
    db.commit()
    db.refresh(session)

    # Runs after the response is sent — the client gets a session id immediately and
    # polls (or listens on the WebSocket) for status updates.
    background_tasks.add_task(run_pipeline_job, session.id)

    return SessionOut.from_model(session)


@router.post("/live", response_model=SessionOut, status_code=201)
@limiter.limit("10/minute")
def create_live_session(
    request: Request,
    payload: LiveSessionIn,
    db: DBSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Creates a session shell for a live video-conference meeting, plus a matching Room
    row (same id) so the WebRTC signaling socket at /ws/rooms/{session_id} (see
    routers/rooms.py) has something to attach to — the room page connects there
    using this session's id as the room id. No audio yet — the room page uploads the
    recorded mix via POST /sessions/{id}/audio once the call ends, which is what
    actually kicks off the pipeline. Status "live" distinguishes this from an
    "uploaded" file-based session while the call is in progress.
    """
    if payload.session_type not in _VALID_SESSION_TYPES:
        raise HTTPException(status_code=400, detail=f"session_type must be one of {_VALID_SESSION_TYPES}")

    session = MeetingSession(
        owner_id=current_user.id, title=payload.title, session_type=payload.session_type,
        status="live", audio_path="",
    )
    db.add(session)
    db.flush()  # assigns session.id without committing, so the Room can reuse it

    room = Room(id=session.id, code=generate_room_code(db), host_id=current_user.id, status="live")
    db.add(room)
    db.commit()
    db.refresh(session)
    return SessionOut.from_model(session)


@router.post("/{session_id}/audio", response_model=SessionOut)
@limiter.limit("20/minute")
async def attach_audio(
    request: Request,
    session_id: str,
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    db: DBSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Called by the room page when a live meeting ends, with the recorded audio mix."""
    session = db.get(MeetingSession, session_id)
    if session is None or session.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found.")

    file_bytes = await audio.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="No audio was recorded — nothing to process.")
    _check_upload_size(file_bytes)

    session.audio_path = save_audio(file_bytes, audio.filename or "meeting.webm", session.id)
    session.status = "uploaded"
    db.commit()
    db.refresh(session)

    background_tasks.add_task(run_pipeline_job, session.id)
    return SessionOut.from_model(session)


@router.get("", response_model=list[SessionListItem])
def list_sessions(db: DBSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    sessions = (
        db.query(MeetingSession)
        .filter(MeetingSession.owner_id == current_user.id)
        .order_by(MeetingSession.created_at.desc())
        .all()
    )
    return [SessionListItem.model_validate(s) for s in sessions]


@router.get("/{session_id}", response_model=SessionOut)
def get_session(
    session_id: str, db: DBSession = Depends(get_db), current_user: User = Depends(get_current_user),
):
    session = db.get(MeetingSession, session_id)
    if session is None or session.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found.")
    return SessionOut.from_model(session)


@router.get("/{session_id}/audio")
def download_audio(
    session_id: str, db: DBSession = Depends(get_db), current_user: User = Depends(get_current_user),
):
    """
    Serves the raw audio back to the frontend for the timestamp-seeking audio player
    (click a quiz question or action item, jump to that moment in the recording).
    """
    from fastapi.responses import FileResponse

    from ..services.storage_service import get_local_path

    session = db.get(MeetingSession, session_id)
    if session is None or session.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found.")
    if not session.audio_path:
        raise HTTPException(status_code=404, detail="Audio is no longer available for this session.")

    local_path = get_local_path(session.audio_path)
    return FileResponse(local_path, media_type="audio/webm")


@router.delete("/{session_id}", status_code=204)
def delete_session(
    session_id: str, db: DBSession = Depends(get_db), current_user: User = Depends(get_current_user),
):
    session = db.get(MeetingSession, session_id)
    if session is None or session.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found.")

    if session.audio_path:
        delete_audio(session.audio_path)

    db.delete(session)  # cascades to ReferenceDocument rows via the FK relationship
    db.commit()
    return None


# ── Q&A ───────────────────────────────────────────────────────────────────────

class AskIn(BaseModel):
    question: str


@router.post("/{session_id}/ask")
@limiter.limit("30/minute")
async def ask_session(
    request: Request,
    session_id: str,
    payload: AskIn,
    db: DBSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    POST /sessions/{session_id}/ask
    Body: { "question": "What did we decide about the deadline?" }

    Streams the answer token by token (Server-Sent Events) so the user
    sees words appearing immediately. Falls back to raw transcript text
    if Pinecone has no vectors for this session.
    """
    session = db.get(MeetingSession, session_id)
    if session is None or session.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found.")
    if session.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Session is not completed yet (status: {session.status}).",
        )
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    raw_text: str | None = None
    if session.transcript_json:
        raw_text = session.transcript_json.get("raw_text")

    from meetmind_ai.qa import answer_question_stream

    def generate():
        try:
            for token in answer_question_stream(session_id, question, raw_text):
                yield f"data: {token}\n\n"
        except Exception as exc:
            yield f"data: [ERROR] {exc}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
