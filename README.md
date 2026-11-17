# MeetMind

**AI meeting intelligence pipeline** — upload a recording or start a live meeting, get a structured transcript with real speaker labels, role-specific summaries, action items, a Q&A chat, and email notifications, all in under a minute.

Built for two audiences: **teams** (action items, role summaries, participant Q&A) and **students** (topic summaries, Bloom's taxonomy quizzes with distractors).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Next.js Frontend                            │
│  Upload flow · Live WebRTC room · Session dashboard · Q&A chat      │
│  Streaming SSE tokens · WebSocket status updates · Audio player     │
└──────────────────────┬──────────────────────────────────────────────┘
                       │ HTTP / WebSocket
┌──────────────────────▼──────────────────────────────────────────────┐
│                       FastAPI Backend                               │
│                                                                     │
│  /sessions  ──► enqueue_pipeline_job()  ──► Redis (Celery broker)   │
│  /sessions/{id}/ask  ──► StreamingResponse (SSE)                    │
│  /sessions/{id}/notify  ──► email_service                           │
│  /ws/rooms/{id}  ──► WebRTC signaling relay                         │
│                                                                     │
│  Auth: JWT + httpOnly cookie · Rate limiting · Audio auto-delete    │
└──────┬───────────────────────────────────────┬───────────────────── ┘
       │ SQLAlchemy                            │ Celery task
       ▼                                       ▼
┌─────────────┐                    ┌──────────────────────────────────┐
│  PostgreSQL │                    │        Celery Worker             │
│             │                    │                                  │
│  Sessions   │                    │  1. Whisper  ──► Transcript      │
│  Users      │                    │  2. WhisperX ──► Speaker labels  │
│  Transcripts│◄───── writes ──────│  3. GPT-4o  ──► Summaries        │
│  Outputs    │                    │  4. GPT-4o  ──► Action items     │
└─────────────┘                    │  5. Pinecone ──► RAG index       │
                                   │  6. GPT-4o  ──► Quiz (Bloom's)   │
       ┌───────────────────────────│                                  │
       │ pub/sub                   │  Retries: 3× exponential backoff │
       ▼                           └──────────────────────────────────┘
┌─────────────┐
│    Redis    │   Celery broker · Result backend
│             │   WebRTC room signaling
│             │   Session status pub/sub
└─────────────┘
```

---

## Features

**AI Pipeline**
- Transcription via OpenAI Whisper API with WhisperX word-level alignment
- Real speaker diarization via WhisperX + pyannote ML model (speaker confidence 0.9) — feature-flagged, heuristic fallback for local dev
- Aspect-based role summaries (Deng et al.) — one summary per speaker, host overview
- Action item extraction with owner, task, and deadline fields
- Bloom's taxonomy quiz generation — 2 LLM calls per section, distractors included
- Hybrid RAG (BM25 + vector) via Pinecone for reference document retrieval during live meetings
- Automated quality evaluation with Ragas in GitHub Actions CI — faithfulness 1.00, response relevancy 0.86

**Backend**
- Async job queue with Celery + Redis — concurrent sessions, 3-retry exponential back-off, task survives worker crashes
- Streaming Q&A chat over completed transcripts via Server-Sent Events
- Action item email notifications per participant via Resend API
- WebRTC mesh video conferencing with Redis-backed signaling
- JWT + httpOnly cookie auth, rate limiting, audio auto-deletion after configurable retention period

**Quality**
- 88 tests across ai-pipeline and backend (mock + real Redis/Postgres)
- Ragas evaluation pipeline runs on every PR — blocks merge on quality regression
- WhisperX integration test validates real speaker separation on audio

---

## Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14 (App Router), TypeScript, Tailwind CSS |
| Backend | FastAPI, SQLAlchemy, Pydantic v2 |
| AI | OpenAI Whisper + GPT-4o-mini, WhisperX, Pinecone |
| Queue | Celery 5, Redis |
| Database | PostgreSQL (SQLite for local dev) |
| Email | Resend API |
| Evaluation | Ragas, GitHub Actions CI |
| Auth | JWT, httpOnly cookies, slowapi rate limiting |
| Infrastructure | Docker Compose, uvicorn |

---

## Quickstart

### Option A — Docker Compose

```bash
cp backend/.env.example backend/.env
# fill in OPENAI_API_KEY and PINECONE_API_KEY
# or leave USE_MOCK_PIPELINE=true to try without API keys

docker compose up --build
```

Frontend: `http://localhost:3000` · API docs: `http://localhost:8000/docs`

### Option B — Run each service directly

```bash
# 1. Redis (required for Celery and WebSocket signaling)
docker run -p 6379:6379 redis:7

# 2. Backend
cd backend
cp .env.example .env        # fill in your keys
pip install -r requirements.txt
pip install "celery[redis]>=5.3.0" resend
uvicorn app.main:app --reload

# 3. Celery worker (new terminal)
cd backend
celery -A app.celery_app worker --loglevel=info -Q pipeline --concurrency=2 --pool=solo

# 4. Frontend (new terminal)
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`, create an account, and either:
- **Upload a recording** — any audio/video file
- **Start a live meeting** — opens a WebRTC room, copy the invite link for others

Once a session completes, the dashboard shows the transcript with click-to-seek, role summaries, action items, and the Q&A chat panel.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for transcription and summarization |
| `PINECONE_API_KEY` | — | Required for RAG (live meeting reference docs) |
| `USE_MOCK_PIPELINE` | `true` | Use realistic mock data without API calls |
| `USE_CELERY` | `false` | Enable Celery queue (requires Redis) |
| `USE_REAL_DIARIZATION` | `false` | Enable WhisperX ML diarization |
| `HUGGINGFACE_TOKEN` | — | Required when USE_REAL_DIARIZATION=true |
| `CELERY_BROKER_URL` | `redis://localhost:6379/0` | Redis broker for Celery |
| `RESEND_API_KEY` | — | Required for action item email notifications |
| `AUDIO_RETENTION_DAYS` | `30` | Days before raw audio is auto-deleted |

---

## Running tests

```bash
# AI pipeline tests (25 tests — all offline)
cd ai-pipeline
pip install -r requirements.txt
python -m pytest tests/ -v

# Ragas evaluation (requires OPENAI_API_KEY)
export OPENAI_API_KEY="sk-..."
python -m pytest evaluation/ragas_eval.py -v -s

# Backend tests (63 tests — all offline except Redis suite)
cd backend
pip install -r requirements.txt
python -m pytest tests/ -v

# WhisperX integration test (requires HF token + audio file)
export RUN_REAL_DIARIZATION_TESTS=1
export TEST_AUDIO_PATH=tests/fixtures/sample_meeting.wav
python -m pytest tests/test_whisperx_diarization.py::test_real_whisperx_on_sample_audio -v -s
```

---

## Project layout

```
meetmind/
├── ai-pipeline/              Python package — no web framework dependency
│   ├── meetmind_ai/
│   │   ├── transcription.py  Whisper API + noise scoring
│   │   ├── diarization.py    Speaker assignment (heuristic or WhisperX)
│   │   ├── real_diarization.py  WhisperX ML diarization
│   │   ├── summarization.py  Aspect-based role summaries (GPT-4o)
│   │   ├── action_items.py   Structured action item extraction
│   │   ├── quiz_generation.py  Bloom's taxonomy quiz (batched LLM)
│   │   ├── rag.py            Pinecone hybrid retrieval
│   │   └── pipeline.py       Orchestrator
│   ├── evaluation/
│   │   ├── ragas_eval.py     Automated quality gate (CI)
│   │   └── test_dataset.json Gold Q&A pairs
│   └── tests/                25 tests
│
├── backend/                  FastAPI app
│   ├── app/
│   │   ├── routers/
│   │   │   ├── sessions.py   Upload, create, list, delete
│   │   │   ├── qa.py         Streaming Q&A (SSE)
│   │   │   └── notify.py     Action item email notifications
│   │   ├── services/
│   │   │   ├── pipeline_service.py  Celery enqueue + job runner
│   │   │   └── email_service.py     Resend integration
│   │   ├── celery_app.py     Celery instance + config
│   │   └── tasks.py          Pipeline Celery task
│   └── tests/                63 tests
│
└── frontend/                 Next.js 14 (App Router)
    ├── app/
    │   ├── components/
    │   │   └── MeetingQA.tsx Streaming chat component
    │   └── lib/api.ts        Typed API client
    └── ...
```

---

## CI

Every pull request to `main` runs:
- **Ragas quality gate** — faithfulness and response relevancy scored against gold dataset, merge blocked on regression
- WhisperX routing tests (7 tests, no GPU needed)
- Backend test suite (63 tests, mocked dependencies)
