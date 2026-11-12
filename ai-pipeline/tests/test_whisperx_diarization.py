"""
Tests for the WhisperX diarization integration.

All tests run WITHOUT needing whisperx, torch, pandas, a GPU, or a HuggingFace
token. They test routing logic and error handling using mocks only, so they run
cleanly in CI alongside the existing mock-mode tests.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meetmind_ai.diarization import diarize, heuristic_diarize
from meetmind_ai.schemas import Transcript, TranscriptSegment


def _make_transcript(n: int = 3) -> Transcript:
    segs = [
        TranscriptSegment(start=float(i * 5), end=float(i * 5 + 4), text=f"Line {i}.")
        for i in range(n)
    ]
    return Transcript(
        session_id="test",
        segments=segs,
        raw_text=" ".join(s.text for s in segs),
    )


# ── routing ───────────────────────────────────────────────────────────────────

def test_heuristic_used_when_flag_off():
    """FLAG off → heuristic runs, confidence = 0.5."""
    with patch("meetmind_ai.diarization.settings") as m:
        m.use_real_diarization = False
        result = diarize(_make_transcript(), audio_file_path="fake.wav")
    assert result.segments[0].speaker_confidence == 0.5


def test_heuristic_used_when_no_audio_path():
    """FLAG on but no audio path → heuristic fallback."""
    with patch("meetmind_ai.diarization.settings") as m:
        m.use_real_diarization = True
        result = diarize(_make_transcript(), audio_file_path=None)
    assert result.segments[0].speaker_confidence == 0.5


def test_real_diarize_called_when_flag_on_and_path_given():
    """FLAG on + audio path → real_diarize() is invoked."""
    with patch("meetmind_ai.diarization.settings") as m, \
         patch("meetmind_ai.real_diarization.real_diarize") as mock_real:
        m.use_real_diarization = True
        mock_real.return_value = _make_transcript()
        diarize(_make_transcript(), audio_file_path="meeting.wav")
    mock_real.assert_called_once()


def test_explicit_diarizer_always_wins():
    """Explicit diarizer overrides the flag and audio path."""
    calls = []

    def fake(transcript):
        calls.append(True)
        return transcript

    with patch("meetmind_ai.diarization.settings") as m:
        m.use_real_diarization = True
        diarize(_make_transcript(), audio_file_path="fake.wav", diarizer=fake)

    assert calls == [True]


# ── error handling ────────────────────────────────────────────────────────────

def test_fails_clearly_without_hf_token():
    """real_diarize gives a clear error when HUGGINGFACE_TOKEN is missing."""
    from meetmind_ai.real_diarization import real_diarize
    with patch("meetmind_ai.real_diarization.settings") as m:
        m.huggingface_token = ""
        with pytest.raises(RuntimeError, match="HUGGINGFACE_TOKEN is not set"):
            real_diarize(_make_transcript(), "fake.wav")


def test_full_pipeline_fails_clearly_without_hf_token():
    """whisperx_transcribe_and_diarize also fails clearly without HF token."""
    from meetmind_ai.real_diarization import whisperx_transcribe_and_diarize
    with patch("meetmind_ai.real_diarization.settings") as m:
        m.huggingface_token = ""
        with pytest.raises(RuntimeError, match="HUGGINGFACE_TOKEN is not set"):
            whisperx_transcribe_and_diarize("fake.wav", "session-1")


# ── WhisperX output mapping (fully mocked, no pandas needed) ─────────────────

def test_speaker_labels_mapped_from_mocked_whisperx():
    """
    Verifies WhisperX diarize output is correctly mapped to MeetMind's
    Transcript schema. Uses a mock that mimics the DataFrame .iterrows() 
    interface without importing pandas.
    """
    from meetmind_ai.real_diarization import real_diarize

    # mock a row that behaves like a pandas Series for dict access
    def make_row(start, end, speaker):
        row = MagicMock()
        row.__getitem__ = lambda self, k: {"start": start, "end": end, "speaker": speaker}[k]
        return (None, row)

    mock_df = MagicMock()
    mock_df.iterrows.return_value = iter([
        make_row(0.0,  4.0,  "SPEAKER_00"),
        make_row(4.5,  9.0,  "SPEAKER_01"),
        make_row(9.5, 14.0,  "SPEAKER_00"),
    ])

    mock_pipeline = MagicMock(return_value=mock_df)
    mock_whisperx = MagicMock()
    mock_whisperx.load_audio.return_value = MagicMock()
    mock_whisperx.DiarizationPipeline.return_value = mock_pipeline

    transcript = Transcript(
        session_id="map-test",
        segments=[
            TranscriptSegment(start=0.0,  end=4.0,  text="Hello from Alice."),
            TranscriptSegment(start=4.5,  end=9.0,  text="Hello from Bob."),
            TranscriptSegment(start=9.5,  end=14.0, text="Alice again."),
        ],
        raw_text="Hello from Alice. Hello from Bob. Alice again.",
    )

    with patch("meetmind_ai.real_diarization.settings") as m, \
         patch("meetmind_ai.real_diarization._get_whisperx", return_value=mock_whisperx):
        m.huggingface_token = "hf_fake"
        result = real_diarize(transcript, "fake.wav")

    assert result.segments[0].speaker == "SPEAKER_00"
    assert result.segments[1].speaker == "SPEAKER_01"
    assert result.segments[2].speaker == "SPEAKER_00"
    assert all(s.speaker_confidence == 0.9 for s in result.segments)


# ── real integration test (skipped by default) ───────────────────────────────

@pytest.mark.skipif(
    not __import__("os").getenv("RUN_REAL_DIARIZATION_TESTS"),
    reason="Set RUN_REAL_DIARIZATION_TESTS=1 to run real WhisperX tests",
)
def test_real_whisperx_on_sample_audio():
    """
    Real integration — needs whisperx installed + HF token + audio file.
    Put a 1-2 min 2-speaker WAV at tests/fixtures/sample_meeting.wav then run:
      $env:RUN_REAL_DIARIZATION_TESTS = "1"
      pytest tests/test_whisperx_diarization.py::test_real_whisperx_on_sample_audio -v -s
    """
    import os
    from meetmind_ai.real_diarization import whisperx_transcribe_and_diarize

    audio_path = os.getenv("TEST_AUDIO_PATH", "tests/fixtures/sample_meeting.wav")
    if not Path(audio_path).exists():
        pytest.skip(f"Audio not found at {audio_path}")

    result = whisperx_transcribe_and_diarize(
        audio_file_path=audio_path,
        session_id="integration-test",
        language="en",
        device="cpu",
    )

    print(f"\nTotal segments: {len(result.segments)}")
    print(f"Speakers found: {set(s.speaker for s in result.segments)}")
    for seg in result.segments[:5]:
        print(f"  [{seg.start:.1f}-{seg.end:.1f}] {seg.speaker}: {seg.text[:60]}")

    assert len(result.segments) > 0
    assert all(s.speaker is not None for s in result.segments)
    assert all(s.speaker_confidence == 0.9 for s in result.segments)
