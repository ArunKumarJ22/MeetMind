from contextlib import asynccontextmanager

import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from .config import settings
from .database import init_db
from .rate_limit import limiter
from .routers import auth, documents, notify, qa, rooms, sessions, websocket
from .services import pubsub
from .services.retention import start_retention_job


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    pubsub.bind_main_loop(asyncio.get_running_loop())
    retention_task = start_retention_job()
    yield
    if retention_task:
        retention_task.cancel()


app = FastAPI(
    title="MeetMind AI API",
    description="Turns lecture and meeting audio into structured, role-specific intelligence.",
    version="0.1.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(sessions.router)
app.include_router(documents.router)
app.include_router(notify.router)
app.include_router(qa.router)  
app.include_router(websocket.router)
app.include_router(rooms.router)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mock_pipeline": settings.use_mock_pipeline,
        "pubsub_backend": pubsub.BACKEND,
        "audio_retention_days": settings.audio_retention_days,
    }
