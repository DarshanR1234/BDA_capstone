import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure UTF-8 stdout and PROJECT_ROOT is on sys.path
sys.stdout.reconfigure(encoding="utf-8")
PROJECT_ROOT = Path(r"D:\fake-news-bda")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup & Shutdown Lifespan Hook.
    Preloads RAG models & indexes into RAM once at startup,
    ensuring sub-second retrieval latency on subsequent user requests.
    """
    print("\n" + "=" * 60)
    print("[FastAPI Backend] Initializing Fake News BDA Service...")
    print("[FastAPI Backend] Preloading RAG hybrid retrieval indexes...")
    
    # Import triggers persistent memory load
    from ai.hybrid_retriever import model, word_matrix, char_matrix
    print("[FastAPI Backend] RAG indexes warm in memory. System READY.")
    print("=" * 60 + "\n")
    
    yield
    
    print("[FastAPI Backend] Shutting down service.")


app = FastAPI(
    title="AI-Powered Fake News Detection API",
    description="Big Data Analytics Capstone — Production-grade RAG pipeline combining hybrid semantic/lexical retrieval with LLM reasoning.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Enable CORS for future frontend (allows all origins during development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles

# Register routes
app.include_router(router)

# Mount frontend web dashboard (React build /dist preferred)
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
elif FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
