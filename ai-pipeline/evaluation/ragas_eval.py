"""
Ragas evaluation for MeetMind's summarization pipeline.

What this tests:
  - For each Q&A pair in test_dataset.json, we build a MeetMind Transcript from
    the reference_context, run aspect_based_role_summaries (the same function the
    real pipeline calls for professional sessions), then ask GPT to answer the
    question using the summary as the retrieved context.
  - Ragas scores each (question, answer, context, ground_truth) tuple on:
      faithfulness      — is the answer grounded in the transcript text? (no hallucination)
      answer_relevancy  — does the answer actually address the question asked?

Why summarization and not rag.retrieve_for_mention:
  retrieve_for_mention requires a live Pinecone index with pre-uploaded documents,
  which makes it impossible to run in CI without real API state. Summarization is
  the core AI output every session produces — evaluating it here gives you a stable,
  reproducible quality gate that runs on every push.

Thresholds (start conservative, raise them as you improve the pipeline):
  faithfulness     >= 0.70
  answer_relevancy >= 0.70

Run locally:
  export OPENAI_API_KEY="sk-..."
  cd ai-pipeline
  pytest evaluation/ragas_eval.py -v -s

CI: .github/workflows/ragas_eval.yml runs this automatically on every PR.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from datasets import Dataset
from openai import OpenAI
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness

# Make meetmind_ai importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meetmind_ai.schemas import SessionType, Transcript, TranscriptSegment
from meetmind_ai.summarization import aspect_based_role_summaries


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_test_dataset() -> list[dict]:
    path = Path(__file__).parent / "test_dataset.json"
    with open(path) as f:
        return json.load(f)


def _build_transcript(session_id: str, raw_text: str) -> Transcript:
    """
    Wraps a plain text string into the Transcript schema MeetMind's pipeline uses.
    We treat the whole reference context as a single segment — no diarization needed
    for evaluation purposes, since we're testing summarization quality, not speaker
    attribution.
    """
    return Transcript(
        session_id=session_id,
        segments=[
            TranscriptSegment(
                start=0.0,
                end=60.0,
                text=raw_text,
                speaker="Participant",
            )
        ],
        raw_text=raw_text,
        noise_score=0.0,
    )


def _ask_question_with_context(question: str, context: str) -> str:
    """
    Calls GPT to answer the question using ONLY the provided context.
    This is the 'answer' Ragas will score for faithfulness and relevancy.
    We use gpt-4o-mini here (cheaper than gpt-4o) since this is a scoring
    step, not the main pipeline.
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the user's question "
                    "using ONLY the information provided in the context below. "
                    "If the context does not contain enough information to answer "
                    "the question, say 'Not found in context.' "
                    "Do not use any outside knowledge.\n\n"
                    f"Context:\n{context}"
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0.0,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


def _get_summary_context(transcript: Transcript) -> str:
    """
    Runs MeetMind's real summarization on the transcript and returns the host
    summary as the retrieved context. This is what Ragas uses as 'contexts'.
    """
    role_summaries = aspect_based_role_summaries(transcript)
    # host_summary is always the first element (see summarization.py line:
    # summaries = [RoleSummary(role="Host", summary=data["host_summary"])])
    host_summary = next(
        (s.summary for s in role_summaries if s.role == "Host"),
        role_summaries[0].summary if role_summaries else transcript.raw_text,
    )
    return host_summary


# ── build the evaluation dataset ─────────────────────────────────────────────

def build_eval_dataset() -> Dataset:
    """
    For each test case:
      1. Build a Transcript from the reference_context string
      2. Run MeetMind's summarization → get the host summary as context
      3. Ask GPT to answer the question using that context
      4. Collect (question, answer, contexts, ground_truth) for Ragas
    """
    test_cases = _load_test_dataset()

    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []

    for i, case in enumerate(test_cases):
        session_id = f"eval-session-{i}"
        transcript = _build_transcript(session_id, case["reference_context"])

        # run the real MeetMind summarization
        summary_context = _get_summary_context(transcript)

        # get a GPT answer grounded in the summary
        answer = _ask_question_with_context(case["question"], summary_context)

        questions.append(case["question"])
        answers.append(answer)
        contexts.append([summary_context])          # Ragas expects list[list[str]]
        ground_truths.append(case["ground_truth"])

        print(f"\n[{i+1}/{len(test_cases)}] Q: {case['question']}")
        print(f"  context (first 120 chars): {summary_context[:120]}...")
        print(f"  answer:  {answer[:120]}")

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })


# ── the pytest test — this is what GitHub Actions runs ───────────────────────


def test_rag_quality():
    dataset = build_eval_dataset()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        scores = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
        )

        results = scores.to_pandas()

        avg_faithfulness = float(results["faithfulness"].mean())
        avg_relevancy = float(results["answer_relevancy"].mean())

        print("\nMeetMind RAG Evaluation Results")

        for _, row in results.iterrows():
            print(f"\nQuestion: {row['question']}")
            print(f"Faithfulness: {row['faithfulness']:.2f}")
            print(f"Answer Relevancy: {row['answer_relevancy']:.2f}")

        print(f"\nAverage Faithfulness: {avg_faithfulness:.2f}")
        print(f"Average Answer Relevancy: {avg_relevancy:.2f}")

        assert avg_faithfulness >= 0.70
        assert avg_relevancy >= 0.70

    finally:
        asyncio.set_event_loop(None)
        loop.close()