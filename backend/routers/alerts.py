from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.models.database import Alert, get_db
from backend.models.schemas import AlertFeedback, AlertOut

router = APIRouter()


@router.get("", response_model=list[AlertOut])
def list_alerts(
    scene: Optional[str] = None,
    reviewed: Optional[bool] = None,
    confirmed: Optional[bool] = None,
    db: Session = Depends(get_db),
):
    q = db.query(Alert)
    if scene:
        q = q.filter(Alert.scene == scene)
    if reviewed is not None:
        q = q.filter(Alert.reviewed == reviewed)
    if confirmed is not None:
        q = q.filter(Alert.confirmed_anomaly == confirmed)
    return q.order_by(Alert.timestamp.desc()).all()


@router.get("/{alert_id}", response_model=AlertOut)
def get_alert(alert_id: int, db: Session = Depends(get_db)):
    row = db.query(Alert).filter(Alert.id == alert_id).first()
    if not row:
        raise HTTPException(404, "Alert not found")
    return row


@router.post("/{alert_id}/feedback")
def feedback(alert_id: int, body: AlertFeedback, db: Session = Depends(get_db)):
    row = db.query(Alert).filter(Alert.id == alert_id).first()
    if not row:
        raise HTTPException(404, "Alert not found")
    row.reviewed = True
    row.confirmed_anomaly = body.confirmed_anomaly
    row.feedback_note = body.feedback_note
    db.commit()
    return {"ok": True}


@router.get("/{alert_id}/clip")
def clip(alert_id: int, db: Session = Depends(get_db)):
    row = db.query(Alert).filter(Alert.id == alert_id).first()
    if not row or not row.clip_path:
        raise HTTPException(404, "Clip not found")
    path = row.clip_path
    if not os.path.isfile(path):
        raise HTTPException(404, "File missing")
    return FileResponse(path, media_type="video/mp4")


@router.get("/{alert_id}/heatmap")
def heatmap(alert_id: int, db: Session = Depends(get_db)):
    row = db.query(Alert).filter(Alert.id == alert_id).first()
    if not row or not row.heatmap_path:
        raise HTTPException(404, "Heatmap not found")
    path = row.heatmap_path
    if not os.path.isfile(path):
        raise HTTPException(404, "File missing")
    return FileResponse(path, media_type="image/png")


@router.delete("/{alert_id}")
def delete_alert(alert_id: int, db: Session = Depends(get_db)):
    row = db.query(Alert).filter(Alert.id == alert_id).first()
    if not row:
        raise HTTPException(404, "Alert not found")
    db.delete(row)
    db.commit()
    return {"ok": True}
