"""
Real diarization using WhisperX (Bain et al., 2022).

WhisperX improves on vanilla Whisper in two ways relevant to MeetMind:
  1. Word-level timestamps via forced alignment (CTC) — vanilla Whisper only gives
     segment-level timestamps, so a 30-second segment all gets one speaker label.
  2. Speaker diarization via pyannote integrated inside WhisperX — but unlike using
     pyannote directly, WhisperX bundles it without the separate HF license gate for
     the diarization model itself (you still need a HF token for the segmentation
     model, but it's one step not two).

Why WhisperX instead of pyannote alone (original real_diarization.py):
  - pyannote alone needs TWO separate HF model license acceptances.
  - WhisperX transcribes + aligns + diarizes in one call, replacing both
    transcription.py's OpenAI API call and the old pyannote diarizer.
  - Runs on CPU (slower) or GPU (fast). No API cost per audio minute.

Setup (one-time, before setting USE_REAL_DIARIZATION=true):
  1. pip install -r requirements-whisperx.txt
  2. Go to https://huggingface.co/pyannote/segmentation-3.0 → accept license
  3. Create HF token at https://huggingface.co/settings/tokens (read access)
  4. Set HUGGINGFACE_TOKEN=hf_... in your .env

The entry point the pipeline calls is `whisperx_transcribe_and_diarize()`.
`real_diarize()` is kept as a thin wrapper for the routing in diarization.py.
"""
from __future__ import annotations

from .config import settings
from .schemas import Transcript, TranscriptSegment
from .transcription import estimate_noise_score

# module-level caches so we don't reload the model on every call
_whisperx_model_cache: dict = {}


def _get_whisperx():
    """Import whisperx lazily so the rest of the app works without it installed."""
    try:
        import whisperx
        return whisperx
    except ImportError as exc:
        raise RuntimeError(
            "whisperx isn't installed. Run:\n"
            "  pip install -r requirements-whisperx.txt\n"
            "from the ai-pipeline/ directory first."
        ) from exc


def _get_model(device: str = "cpu", compute_type: str = "int8"):
    """
    Load and cache the WhisperX model. Uses 'base' by default — fast enough for
    demos and accurate for English. Switch to 'large-v2' for production quality.
    Model choices: tiny, base, small, medium, large-v1, large-v2, large-v3
    """
    global _whisperx_model_cache
    key = (device, compute_type)
    if key not in _whisperx_model_cache:
        whisperx = _get_whisperx()
        _whisperx_model_cache[key] = whisperx.load_model(
            "base",
            device=device,
            compute_type=compute_type,
        )
    return _whisperx_model_cache[key]


def whisperx_transcribe_and_diarize(
    audio_file_path: str,
    session_id: str,
    language: str = "en",
    device: str = "cpu",
) -> Transcript:
    """
    Full pipeline: transcription + word-level alignment + speaker diarization,
    all in one function. Replaces both transcription.py's OpenAI call and the
    old pyannote-based diarizer when USE_REAL_DIARIZATION=true.

    Returns a Transcript with:
      - Accurate word-level aligned timestamps (not Whisper's approximate segments)
      - Real speaker labels (SPEAKER_00, SPEAKER_01, ...) from diarization
      - speaker_confidence=0.9 (WhisperX's diarization is model-based, not heuristic)
    """
    if not settings.huggingface_token:
        raise RuntimeError(
            "HUGGINGFACE_TOKEN is not set. WhisperX needs a Hugging Face read token\n"
            "to download the speaker segmentation model.\n"
            "  1. Go to https://huggingface.co/settings/tokens and create a token\n"
            "  2. Accept the license at https://huggingface.co/pyannote/segmentation-3.0\n"
            "  3. Add HUGGINGFACE_TOKEN=hf_... to your .env file"
        )

    whisperx = _get_whisperx()

    # ── Step 1: transcribe with WhisperX ─────────────────────────────────────
    compute_type = "float32" if device == "cpu" else "float16"
    model = _get_model(device=device, compute_type=compute_type)

    audio = whisperx.load_audio(audio_file_path)
    result = model.transcribe(audio, batch_size=4, language=language)

    # ── Step 2: forced alignment for word-level timestamps ───────────────────
    # This is the key WhisperX improvement — vanilla Whisper segments can be 30s long
    # with a single timestamp. Alignment gives us per-word start/end times.
    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=device,
    )
    aligned = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        device=device,
        return_char_alignments=False,
    )

    # ── Step 3: speaker diarization ──────────────────────────────────────────
    from whisperx.diarize import DiarizationPipeline
    diarize_model = DiarizationPipeline(
        token=settings.huggingface_token,
        device=device,
    )
    diarize_segments = diarize_model(audio)

    # assign speaker labels to each word based on overlap
    result_with_speakers = whisperx.assign_word_speakers(diarize_segments, aligned)

    # ── Step 4: convert WhisperX output → MeetMind Transcript ────────────────
    segments: list[TranscriptSegment] = []
    for seg in result_with_speakers["segments"]:
        # WhisperX puts speaker on the segment after assign_word_speakers
        speaker = seg.get("speaker", "SPEAKER_00")

        segments.append(TranscriptSegment(
            start=float(seg["start"]),
            end=float(seg["end"]),
            text=seg["text"].strip(),
            speaker=speaker,
            speaker_confidence=0.9,  # model-based, not heuristic
        ))

    raw_text = " ".join(s.text for s in segments)

    return Transcript(
        session_id=session_id,
        language=result.get("language", language),
        segments=segments,
        raw_text=raw_text,
        noise_score=estimate_noise_score(raw_text),
    )


def real_diarize(transcript: Transcript, audio_file_path: str) -> Transcript:
    """
    Thin wrapper kept for backward compatibility with diarization.py's routing.
    Called when USE_REAL_DIARIZATION=true and an audio_file_path is provided.

    Note: when called from pipeline.py via run_pipeline(), the transcript already
    exists (from the OpenAI Whisper API call). This applies WhisperX diarization
    to that existing transcript's segments rather than re-transcribing.
    For full WhisperX quality (word-level alignment), call
    whisperx_transcribe_and_diarize() directly instead of run_pipeline().
    """
    if not settings.huggingface_token:
        raise RuntimeError(
            "HUGGINGFACE_TOKEN is not set. See real_diarization.py docstring for setup."
        )

    whisperx = _get_whisperx()

    device = "cpu"
    audio = whisperx.load_audio(audio_file_path)

    # diarize the audio to get speaker turn boundaries
    from whisperx.diarize import DiarizationPipeline
    diarize_model = DiarizationPipeline(
        token=settings.huggingface_token,
        device=device,
    )
    diarize_segments = diarize_model(audio)

    # build turn list: [(start, end, speaker_label), ...]
    turns: list[tuple[float, float, str]] = []
    for _, row in diarize_segments.iterrows():
        turns.append((float(row["start"]), float(row["end"]), str(row["speaker"])))

    def _speaker_for(seg_start: float, seg_end: float) -> str:
        """Assign the speaker whose turn overlaps this segment the most."""
        best_label, best_overlap = "SPEAKER_00", 0.0
        for t_start, t_end, label in turns:
            overlap = max(0.0, min(seg_end, t_end) - max(seg_start, t_start))
            if overlap > best_overlap:
                best_overlap, best_label = overlap, label
        return best_label

    new_segments = [
        seg.model_copy(update={
            "speaker": _speaker_for(seg.start, seg.end),
            "speaker_confidence": 0.9,
        })
        for seg in transcript.segments
    ]

    return transcript.model_copy(update={"segments": new_segments})