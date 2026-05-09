from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.models.database import Alert, get_db
from backend.models.schemas import (
    ActiveModelBody,
    ModeBody,
    SensitivityBody,
    SystemStatus,
    ThresholdBody,
)
from backend.services import state as app_state

router = APIRouter()


def _list_scenes() -> list[str]:
    root = Path("data/drone")
    if not root.is_dir():
        return []
    return sorted(
        [p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    )


def _estimate_harvest_hours() -> float:
    fps = 30.0
    n_frames = 0
    root = Path("data/drone")
    if not root.is_dir():
        return 0.0
    for scene in root.iterdir():
        if not scene.is_dir():
            continue
        frames_root = scene / "training" / "frames"
        if not frames_root.is_dir():
            continue
        for vid in frames_root.iterdir():
            if vid.is_dir():
                n_frames += len(list(vid.glob("*.jpg")))
    return round(n_frames / fps / 3600.0, 2)


@router.get("/status", response_model=SystemStatus)
def get_status(db: Session = Depends(get_db)):
    scenes = _list_scenes()
    total = db.query(Alert).count()
    unrev = db.query(Alert).filter(Alert.reviewed.is_(False)).count()
    return SystemStatus(
        mode=app_state.get_mode(),
        active_model=app_state.get_active_model(),
        scenes_available=scenes,
        total_alerts=total,
        unreviewed_alerts=unrev,
        harvest_hours=_estimate_harvest_hours(),
    )


@router.post("/mode")
def set_mode(body: ModeBody):
    app_state.set_mode(body.mode)
    return {"ok": True, "mode": body.mode}


@router.get("/scenes")
def scenes():
    return {"scenes": _list_scenes()}


@router.post("/active-model")
def active_model(body: ActiveModelBody):
    app_state.set_active_model(body.model)
    return {"ok": True, "active_model": body.model}


@router.post("/threshold")
def threshold(body: ThresholdBody):
    app_state.set_threshold(body.model_type, body.scene, body.threshold)
    return {"ok": True}


@router.post("/sensitivity")
def sensitivity(body: SensitivityBody):
    pct = {"low": 99, "medium": 95, "high": 90}.get(body.level.lower(), 95)
    app_state.THRESHOLD_PERCENTILE = pct
    return {"ok": True, "threshold_percentile": pct}
