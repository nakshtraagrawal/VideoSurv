from __future__ import annotations

import os
import shutil
import threading
import uuid
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.jobs import create_job, get_job, update_job
from backend.models.database import Alert, SessionLocal
from backend.models.schemas import InferRequest
from backend.services import state as app_state
from backend.services.detector import detector
from pipeline import preprocess

router = APIRouter()


def _safe_frames_dir(path: str) -> str:
    p = Path(path).resolve()
    if not p.is_dir():
        raise HTTPException(400, "frames_dir not found")
    return str(p)


@router.post("")
def infer_start(body: InferRequest):
    frames_dir = _safe_frames_dir(body.frames_dir)
    model_type = body.model_type
    scene = body.scene
    thr = body.threshold
    if thr is None:
        thr = app_state.get_threshold(model_type, scene)
    job = create_job("infer")

    def _run():
        db = SessionLocal()
        try:
            update_job(job.id, status="running")
            scores, out_mp4 = detector.score_video_folder(
                frames_dir, model_type, scene, thr
            )
            mx = float(np.max(scores)) if scores else 0.0
            if mx > thr:
                name = Path(frames_dir).name
                al = Alert(
                    scene=scene,
                    video_name=name,
                    frame_idx=int(np.argmax(scores)) if scores else 0,
                    anomaly_score=mx,
                    model_used=model_type,
                    clip_path=out_mp4,
                    heatmap_path=None,
                    reviewed=False,
                    confirmed_anomaly=None,
                )
                db.add(al)
                db.commit()
            update_job(
                job.id,
                status="done",
                result={
                    "scores_path": out_mp4.replace(".mp4", "_scores.npy"),
                    "video_path": out_mp4,
                    "max_score": mx,
                },
            )
        except Exception as e:
            update_job(job.id, status="failed", error=str(e))
        finally:
            db.close()

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job.id}


@router.get("/{job_id}")
def infer_status(job_id: str):
    j = get_job(job_id)
    if not j:
        raise HTTPException(404, "Unknown job")
    return {
        "id": j.id,
        "kind": j.kind,
        "status": j.status,
        "result": j.result,
        "error": j.error,
    }


@router.post("/upload")
async def infer_upload(
    file: UploadFile = File(...),
    model_type: str = Form(default="hstforu"),
    scene: str = Form(default="bike"),
):
    os.makedirs("uploads", exist_ok=True)
    uid = str(uuid.uuid4())[:8]
    dest = Path("uploads") / f"{uid}_{file.filename}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    frames_root = Path("uploads") / f"{uid}_frames"
    preprocess.extract_frames(str(dest), str(frames_root), size=256)
    job = create_job("infer_upload")

    def _run():
        db = SessionLocal()
        try:
            update_job(job.id, status="running")
            thr = app_state.get_threshold(model_type, scene)
            scores, out_mp4 = detector.score_video_folder(
                str(frames_root), model_type, scene, thr
            )
            mx = float(np.max(scores)) if scores else 0.0
            if mx > thr:
                al = Alert(
                    scene=scene,
                    video_name=dest.name,
                    frame_idx=int(np.argmax(scores)) if scores else 0,
                    anomaly_score=mx,
                    model_used=model_type,
                    clip_path=out_mp4,
                    heatmap_path=None,
                    reviewed=False,
                    confirmed_anomaly=None,
                )
                db.add(al)
                db.commit()
            update_job(
                job.id,
                status="done",
                result={"video_path": out_mp4, "upload": str(dest)},
            )
        except Exception as e:
            update_job(job.id, status="failed", error=str(e))
        finally:
            db.close()

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job.id, "frames_dir": str(frames_root)}
