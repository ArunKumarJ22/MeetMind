"""
Summarization stage, shared by both output branches:

- Student branch uses `segment_by_topic` (Liu et al., 2021, dynamic sliding window) to
  produce topic-organized, timestamped sections.
- Professional branch uses `aspect_based_role_summaries` (Deng et al., 2023) to split
  the transcript into decision / action-item / background / discussion sentences and
  generate a full summary for the host plus a personalized slice per participant.
"""
from __future__ import annotations

import json

from openai import OpenAI

from .config import settings
from .schemas import ConceptSection, RoleSummary, Transcript

_SEGMENTATION_PROMPT = """You segment a meeting/lecture transcript into topic-coherent \
sections, the way a human would notice the conversation moving from one subject to the \
next (this mirrors Liu et al.'s dynamic sliding window approach — respect topic \
boundaries, don't chunk by a fixed size).

Transcript (each line is "[start-end] speaker: text"):
{lines}

Return ONLY a JSON array. Each element:
{{"title": short topic title, "start": number, "end": number, "summary": 2-3 sentence \
summary in your own words, "key_concepts": [3-6 short phrases]}}
Cover the entire transcript with contiguous, non-overlapping sections ordered by time."""

_ASPECT_PROMPT = """Classify this meeting transcript's content into a full summary for \
the meeting host, and a short personalized summary for each named speaker containing \
only what is relevant to them (their commitments, mentions of their name, decisions \
that affect them) — this mirrors Deng et al.'s aspect-based classification into \
decision / action-item / background / discussion sentences.

Rules:
- In the host_summary, always refer to participants by their exact name from the transcript.
  Never replace names with "a participant", "one team member", or pronouns.
- Include every named speaker, their specific tasks, commitments, and deadlines.
- Only include facts explicitly stated in the transcript. Do not infer anything.

Transcript (each line is "[start-end] speaker: text"):
{lines}

Return ONLY a JSON object:
{{"host_summary": string, "participant_summaries": [{{"role": speaker name, "summary": string}}]}}"""


def _format_lines(transcript: Transcript) -> str:
    return "\n".join(
        f"[{s.start:.1f}-{s.end:.1f}] {s.speaker or 'Unknown'}: {s.text}"
        for s in transcript.segments
    )


def segment_by_topic(transcript: Transcript) -> list[ConceptSection]:
    settings.require_openai()
    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{
            "role": "user",
            "content": _SEGMENTATION_PROMPT.format(lines=_format_lines(transcript)),
        }],
        temperature=0.2,
        # No response_format here: OpenAI's json_object mode requires a top-level
        # object, but segmentation naturally returns an array. We strip markdown
        # fences defensively below instead.
    )
    raw = resp.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(raw)
    return [ConceptSection(**item) for item in data]


def aspect_based_role_summaries(transcript: Transcript) -> list[RoleSummary]:
    settings.require_openai()
    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{
            "role": "user",
            "content": _ASPECT_PROMPT.format(lines=_format_lines(transcript)),
        }],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    summaries = [RoleSummary(role="Host", summary=data["host_summary"])]
    summaries += [RoleSummary(**p) for p in data.get("participant_summaries", [])]
    return summaries
