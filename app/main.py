from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.api.live_ws import router as live_ws_router
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

from contextlib import asynccontextmanager
from app.services.llm_queue import llm_queue

@asynccontextmanager
async def lifespan(app: FastAPI):
    llm_queue.start()
    yield
    llm_queue.stop()

app = FastAPI(
    title="Corvus LLM Service",
    description="Microservicio de inferencia IA para evaluación de propuestas académicas con sesiones conversacionales.",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1/llm")
app.include_router(live_ws_router, prefix="/api/v1/llm")
