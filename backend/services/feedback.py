"""Feedback ingestion and retraining bookkeeping."""
from __future__ import annotations

from sqlalchemy.orm import Session

from backend.models.database import Alert, TrainingRun
from backend.services import state as app_state


def count_feedback_normal(db: Session) -> int:
    return (
        db.query(Alert)
        .filter(Alert.reviewed.is_(True))
        .filter(Alert.confirmed_anomaly.is_(False))
        .count()
    )


def queue_retrain_from_feedback(db: Session) -> TrainingRun | None:
    n = count_feedback_normal(db)
    if n < 10:
        return None
    run = TrainingRun(
        model_type=app_state.get_active_model(),
        scene="feedback",
        status="queued",
        final_loss=None,
        auc=None,
        checkpoint_path=None,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run
