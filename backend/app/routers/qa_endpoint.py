"""
Q&A endpoint — add these lines to backend/app/routers/sessions.py

INSTRUCTIONS:
  1. Add the import at the top of sessions.py (with the other imports):
       from fastapi.responses import StreamingResponse

  2. Copy the AskIn class and ask_session function below into sessions.py,
     after the existing delete_session route.

  3. That's it. The endpoint is automatically registered because sessions.router
     is already included in main.py.
"""

# ── Paste this import at the top of sessions.py ──────────────────────────────
# from fastapi.responses import StreamingResponse

# ── Paste these two blocks after delete_session in sessions.py ───────────────

from pydantic import BaseModel


class AskIn(BaseModel):
    question: str


# @router.post("/{session_id}/ask")
# @limiter.limit("30/minute")
# async def ask_session(
#     request: Request,
#     session_id: str,
#     payload: AskIn,
#     db: DBSession = Depends(get_db),
#     current_user: User = Depends(get_current_user),
# ):
#     """
#     POST /sessions/{session_id}/ask
#     Body: { "question": "What did we decide about the deadline?" }
#
#     Streams the answer token by token using Server-Sent Events so the user
#     sees words appearing immediately rather than waiting for the full response.
#
#     Returns 404 if the session doesn't exist or isn't completed yet.
#     Returns 400 if the question is empty.
#     """
#     session = db.get(MeetingSession, session_id)
#     if session is None or session.owner_id != current_user.id:
#         raise HTTPException(status_code=404, detail="Session not found.")
#     if session.status != "completed":
#         raise HTTPException(
#             status_code=400,
#             detail=f"Session is not completed yet (status: {session.status}). "
#                    "Wait for processing to finish before asking questions.",
#         )
#     question = payload.question.strip()
#     if not question:
#         raise HTTPException(status_code=400, detail="Question cannot be empty.")
#
#     # pull raw transcript text as fallback if Pinecone has no chunks
#     raw_text: str | None = None
#     if session.transcript_json:
#         raw_text = session.transcript_json.get("raw_text")
#
#     from meetmind_ai.qa import answer_question_stream
#
#     def generate():
#         try:
#             for token in answer_question_stream(session_id, question, raw_text):
#                 # SSE format: "data: <token>\n\n"
#                 yield f"data: {token}\n\n"
#         except Exception as exc:
#             yield f"data: [ERROR] {exc}\n\n"
#         finally:
#             yield "data: [DONE]\n\n"
#
#     return StreamingResponse(
#         generate(),
#         media_type="text/event-stream",
#         headers={
#             "Cache-Control": "no-cache",
#             "X-Accel-Buffering": "no",   # nginx: disable response buffering
#         },
#     )
