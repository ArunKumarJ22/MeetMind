"""
Ragas evaluation for MeetMind's summarization pipeline.
Written for Ragas >= 0.3 (the new API with llm_factory, embedding_factory,
instantiated metric objects, EvaluationDataset, and new column names).

Run locally:
  $env:OPENAI_API_KEY = "sk-..."          # Windows PowerShell
  cd ai-pipeline
  python -m pytest evaluation/ragas_eval.py -v -s
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings

# ── Ragas >= 0.3 imports ─────────────────────────────────────────────────────
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.llms import llm_factory
from ragas.metrics import Faithfulness

try:
    from ragas.metrics import ResponseRelevancy
except ImportError:  # pragma: no cover - compatibility with older Ragas releases
    from ragas.metrics import AnswerRelevancy as ResponseRelevancy

from meetmind_ai.schemas import Transcript, TranscriptSegment
from meetmind_ai.summarization import aspect_based_role_summaries


# ── helpers ───────────────────────────────────────────────────────────────────

def _resolve_relevancy_column(results) -> str:
    """Support both Ragas naming schemes: response_relevancy and answer_relevancy."""
    for candidate in ("response_relevancy", "answer_relevancy"):
        if candidate in results.columns:
            return candidate

    relevancy_like = [
        column
        for column in results.columns
        if "relev" in str(column).lower()
    ]
    if relevancy_like:
        return relevancy_like[0]

    raise KeyError(
        "Unable to find a relevancy column in the Ragas evaluation output. "
        f"Available columns: {list(results.columns)}"
    )


def _load_test_dataset() -> list[dict]:
    path = Path(__file__).parent / "test_dataset.json"
    with open(path) as f:
        return json.load(f)


def _build_transcript(session_id: str, segments_data: list[dict]) -> Transcript:
    """Build a real MeetMind Transcript with named speaker segments."""
    segments = [
        TranscriptSegment(
            start=s["start"],
            end=s["end"],
            text=s["text"],
            speaker=s["speaker"],
        )
        for s in segments_data
    ]
    return Transcript(
        session_id=session_id,
        segments=segments,
        raw_text=" ".join(s.text for s in segments),
        noise_score=0.0,
    )


def _ask_question_with_context(question: str, context: str, client: OpenAI) -> str:
    """Ask GPT to answer using ONLY the provided summary context."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the user's question "
                    "using ONLY the information in the context below. "
                    "If the answer is not in the context, say exactly: "
                    "'Not found in context.'\n"
                    "Always answer in a complete sentence that restates the question topic. "
                    "For example: 'The deadline for X is Friday' not just 'Friday'.\n\n"
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
    Run MeetMind's real summarization and return host + participant summaries
    concatenated so speaker-specific questions are answerable.
    """
    role_summaries = aspect_based_role_summaries(transcript)
    host = next(
        (s.summary for s in role_summaries if s.role == "Host"),
        role_summaries[0].summary if role_summaries else transcript.raw_text,
    )
    participant_text = "\n".join(
        f"{s.role}: {s.summary}"
        for s in role_summaries
        if s.role != "Host"
    )
    return f"{host}\n\nParticipant summaries:\n{participant_text}"


# ── build the EvaluationDataset (Ragas >= 0.3 format) ────────────────────────

def build_eval_dataset(openai_client: OpenAI) -> EvaluationDataset:
    test_cases = _load_test_dataset()
    samples: list[SingleTurnSample] = []

    for i, case in enumerate(test_cases):
        transcript = _build_transcript(f"eval-{i}", case["segments"])
        context = _get_summary_context(transcript)
        answer = _ask_question_with_context(case["question"], context, openai_client)

        print(f"\n[{i+1}/{len(test_cases)}] Q: {case['question']}")
        print(f"  context (first 200 chars): {context[:200]}...")
        print(f"  answer: {answer}")

        samples.append(SingleTurnSample(
            user_input=case["question"],
            response=answer,
            retrieved_contexts=[context],
            reference=case["ground_truth"],
        ))

    return EvaluationDataset(samples=samples)


# ── the pytest test ───────────────────────────────────────────────────────────

def test_rag_quality():
    """
    Runs MeetMind's summarization on the gold dataset and asserts that
    faithfulness and response relevancy meet minimum thresholds.

    Thresholds:
      faithfulness       >= 0.70  (no hallucination)
      response_relevancy >= 0.70  (answer addresses the question)
    """
    api_key = os.environ["OPENAI_API_KEY"]
    openai_client = OpenAI(api_key=api_key)

    # Ragas expects LangChain-style embeddings for the relevancy metric,
    # not the ragas openai provider object.
    ragas_llm = llm_factory(model="gpt-4o-mini", client=openai_client)
    ragas_embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
    )

    # metrics must be instantiated objects in Ragas >= 0.3
    metrics = [
        Faithfulness(llm=ragas_llm),
        ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
    ]

    dataset = build_eval_dataset(openai_client)

    scores = evaluate(
        dataset=dataset,
        metrics=metrics,
    )

    results = scores.to_pandas()

    # Ragas output names vary by release: "response_relevancy" and
    # "answer_relevancy" are both used in the wild.
    faith_col = "faithfulness"
    rel_col = _resolve_relevancy_column(results)
    q_col = "user_input"

    avg_faithfulness = float(results[faith_col].mean())
    avg_relevancy = float(results[rel_col].mean())

    print("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  MeetMind RAG Evaluation Results")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    for _, row in results.iterrows():
        print(f"\n  Q: {row[q_col]}")
        print(f"     faithfulness:       {row[faith_col]:.2f}")
        print(f"     relevancy:          {row[rel_col]:.2f}")
    print(f"\n  Average faithfulness: {avg_faithfulness:.2f}  (threshold ≥ 0.70)")
    print(f"  Average relevancy:    {avg_relevancy:.2f}  (threshold ≥ 0.70)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    assert avg_faithfulness >= 0.70, (
        f"Faithfulness {avg_faithfulness:.2f} below 0.70. "
        "The summarization is hallucinating. See per-question scores above."
    )
    assert avg_relevancy >= 0.70, (
        f"Response relevancy {avg_relevancy:.2f} below 0.70. "
        "The summary is losing key facts. See per-question scores above."
    )
