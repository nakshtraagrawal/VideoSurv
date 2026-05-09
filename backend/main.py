"""FastAPI entrypoint"""
import glob
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routers import alerts, benchmark, inference, scores, system, training
from backend.services.detector import detector


@asynccontextmanager
async def lifespan(app: FastAPI):
    for ckpt in glob.glob("checkpoints/*/best.pth"):
        try:
            detector.load_from_checkpoint_path(ckpt)
        except Exception as e:
            print(f"Failed to load {ckpt}: {e}")
    yield


app = FastAPI(title="AUTOSURVEIL API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system.router, prefix="/api/system", tags=["system"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["alerts"])
app.include_router(inference.router, prefix="/api/infer", tags=["infer"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(benchmark.router, prefix="/api/benchmark", tags=["benchmark"])
app.include_router(scores.router, prefix="/api", tags=["scores"])

os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


@app.get("/health")
def health():
    return {"status": "ok"}
