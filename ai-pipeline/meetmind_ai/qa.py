"""
Layer 7 — Post-meeting Q&A (ask anything about a completed session).

Design:
  The user asks a free-form question about a completed meeting.
  We embed the question, retrieve the top-k most relevant transcript chunks
  from Pinecone (same index used by retrieve_for_mention), then ask GPT to
  answer using ONLY those chunks — no hallucination of facts not in the transcript.

  If Pinecone has no chunks for the session (e.g. the session was processed
  before this feature existed), we fall back to the raw transcript text stored
  in PostgreSQL, chunking it on the fly.

Why not use the existing retrieve_for_mention:
  retrieve_for_mention is designed for LIVE use — it queries by rolling timestamp
  window. Q&A is POST-meeting — we want global retrieval across the whole transcript,
  not just a time window.

Streaming:
  answer_question_stream() is a generator that yields tokens as they arrive from
  OpenAI. The FastAPI endpoint uses StreamingResponse to push them to the browser.
  This means the user sees words appearing immediately instead of waiting 10-30s
  for the full answer.
"""
from __future__ import annotations

from typing import Generator

from openai import OpenAI

from .config import settings
from .rag import _embed, _pinecone_index

_QA_SYSTEM_PROMPT = """You are MeetMind, an AI assistant that answers questions about \
meeting transcripts.

Rules:
- Answer using ONLY the transcript excerpts provided in the context below.
- If the answer is not in the excerpts, say: "I couldn't find that in the transcript."
- Always cite WHICH speaker said what, if the speaker is known.
- Be concise. One to three sentences unless the question requires more detail.
- Never invent facts, names, dates, or decisions not explicitly in the context.

Context (transcript excerpts, most relevant first):
{context}"""


def _build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context block for the LLM."""
    if not chunks:
        return "(no relevant excerpts found)"
    parts = []
    for i, chunk in enumerate(chunks, 1):
        speaker = chunk.get("speaker", "")
        text = chunk.get("text", "")
        ts = chunk.get("timestamp", "")
        speaker_label = f"[{speaker}] " if speaker else ""
        time_label = f"({ts}) " if ts else ""
        parts.append(f"[{i}] {time_label}{speaker_label}{text}")
    return "\n\n".join(parts)


def _retrieve_chunks(session_id: str, question: str, top_k: int = 5) -> list[dict]:
    """
    Embed the question and retrieve the top_k most similar transcript chunks
    from Pinecone for this session.
    Returns a list of dicts with keys: text, speaker, timestamp, score.
    """
    try:
        index = _pinecone_index()
        query_vector = _embed(question)
        result = index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=session_id,
            include_metadata=True,
        )
        chunks = []
        for match in result.get("matches", []):
            meta = match.get("metadata", {})
            chunks.append({
                "text": meta.get("text", ""),
                "speaker": meta.get("speaker", ""),
                "timestamp": meta.get("timestamp", ""),
                "score": match.get("score", 0.0),
            })
        return chunks
    except Exception:
        # Pinecone unavailable or index empty — caller will fall back to raw transcript
        return []


def _chunk_raw_transcript(raw_text: str, chunk_size: int = 400) -> list[dict]:
    """
    Fallback: split raw transcript text into overlapping chunks when Pinecone
    has no vectors for this session. Used when the session was processed before
    transcript chunks were indexed.
    """
    words = raw_text.split()
    chunks = []
    step = chunk_size // 2  # 50% overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk.strip():
            chunks.append({"text": chunk, "speaker": "", "timestamp": "", "score": 1.0})
    return chunks[:5]  # cap at 5 like Pinecone top_k


def answer_question_stream(
    session_id: str,
    question: str,
    transcript_raw_text: str | None = None,
) -> Generator[str, None, None]:
    """
    Main entry point. Retrieves relevant transcript chunks, builds a prompt,
    and streams GPT's answer token by token.

    Args:
        session_id: used as the Pinecone namespace to scope retrieval.
        question: the user's free-form question.
        transcript_raw_text: fallback text if Pinecone has no chunks for this session.

    Yields:
        str tokens as they arrive from OpenAI.
    """
    settings.require_openai()
    client = OpenAI(api_key=settings.openai_api_key)

    # 1. Try Pinecone retrieval first
    chunks = _retrieve_chunks(session_id, question)

    # 2. Fall back to raw transcript chunking if Pinecone returned nothing
    if not chunks and transcript_raw_text:
        chunks = _chunk_raw_transcript(transcript_raw_text)

    context = _build_context(chunks)
    system_prompt = _QA_SYSTEM_PROMPT.format(context=context)

    # 3. Stream the answer
    stream = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.2,   # low temp = factual, grounded answers
        max_tokens=500,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


def answer_question(
    session_id: str,
    question: str,
    transcript_raw_text: str | None = None,
) -> str:
    """
    Non-streaming version — returns the full answer as a single string.
    Used by tests and the Ragas evaluation suite.
    """
    return "".join(answer_question_stream(session_id, question, transcript_raw_text))
