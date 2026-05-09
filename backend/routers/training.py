from __future__ import annotations

import datetime
import subprocess
import sys
import threading
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.models.database import SessionLocal, TrainingRun, get_db
from backend.models.schemas import TrainingRequest, TrainingRunOut
from backend.services import feedback as feedback_service
from backend.services import state as app_state

router = APIRouter()


def _scene_path(scene: str) -> str:
    p = Path(scene)
    if p.is_dir():
        return str(p.resolve())
    return str((Path("data/drone") / scene).resolve())


def _train_script(model_type: str) -> str:
    if model_type == "conv_ae":
        return "models/conv_autoencoder/train.py"
    if model_type == "astnet":
        return "models/astnet/train.py"
    if model_type == "hstforu":
        return "models/hstforu/train.py"
    raise ValueError(model_type)


@router.post("/start", response_model=TrainingRunOut)
def start_training(body: TrainingRequest, db: Session = Depends(get_db)):
    try:
        script = _train_script(body.model_type)
    except ValueError:
        raise HTTPException(400, "Invalid model_type")
    scene_path = _scene_path(body.scene)
    if not Path(scene_path).is_dir():
        raise HTTPException(400, f"Scene path not found: {scene_path}")

    run = TrainingRun(
        model_type=body.model_type,
        scene=Path(scene_path).name,
        status="queued",
        start_time=datetime.datetime.utcnow(),
        end_time=None,
        final_loss=None,
        auc=None,
        checkpoint_path=None,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    epochs = body.epochs

    def _proc():
        db2 = SessionLocal()
        try:
            row = db2.query(TrainingRun).filter(TrainingRun.id == run.id).first()
            if row:
                row.status = "running"
                db2.commit()
            cmd = [sys.executable, script, "--scene", scene_path]
            if epochs is not None:
                cmd.extend(["--epochs", str(epochs)])
            r = subprocess.run(cmd, capture_output=True, text=True)
            row = db2.query(TrainingRun).filter(TrainingRun.id == run.id).first()
            if not row:
                return
            ckpt = Path("checkpoints") / f"{body.model_type}_{Path(scene_path).name}" / "best.pth"
            row.checkpoint_path = str(ckpt) if ckpt.is_file() else None
            row.end_time = datetime.datetime.utcnow()
            if r.returncode == 0:
                row.status = "complete"
            else:
                row.status = "failed"
            db2.commit()
        finally:
            db2.close()

    threading.Thread(target=_proc, daemon=True).start()
    return run


@router.get("/{run_id}", response_model=TrainingRunOut)
def get_run(run_id: int, db: Session = Depends(get_db)):
    row = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not row:
        raise HTTPException(404, "Not found")
    return row


@router.get("/history", response_model=list[TrainingRunOut])
def history(db: Session = Depends(get_db)):
    return db.query(TrainingRun).order_by(TrainingRun.id.desc()).limit(100).all()


@router.post("/retrain")
def retrain(db: Session = Depends(get_db)):
    run = feedback_service.queue_retrain_from_feedback(db)
    if run is None:
        return {"queued": False, "message": "Not enough feedback samples (need 10+)"}
    return {"queued": True, "training_run_id": run.id}
