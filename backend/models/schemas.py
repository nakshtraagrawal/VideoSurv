from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict

_SCHEMA_CFG = ConfigDict(from_attributes=True, protected_namespaces=())


class AlertOut(BaseModel):
    model_config = _SCHEMA_CFG

    id: int
    scene: str
    video_name: str
    frame_idx: int
    anomaly_score: float
    model_used: str
    clip_path: Optional[str]
    heatmap_path: Optional[str]
    timestamp: datetime
    reviewed: bool
    confirmed_anomaly: Optional[bool]


class AlertFeedback(BaseModel):
    confirmed_anomaly: bool
    feedback_note: Optional[str] = None


class TrainingRequest(BaseModel):
    model_type: str
    scene: str
    epochs: Optional[int] = None


class TrainingRunOut(BaseModel):
    model_config = _SCHEMA_CFG

    id: int
    model_type: str
    scene: str
    status: str
    final_loss: Optional[float]
    auc: Optional[float]


class BenchmarkResultOut(BaseModel):
    model_config = _SCHEMA_CFG

    model_type: str
    scene: str
    auc: float
    precision: float
    recall: float
    fpr: float
    threshold: float


class InferRequest(BaseModel):
    frames_dir: str
    model_type: str
    scene: str
    threshold: Optional[float] = None


class SystemStatus(BaseModel):
    mode: str
    active_model: str
    scenes_available: list[str]
    total_alerts: int
    unreviewed_alerts: int
    harvest_hours: float


class ModeBody(BaseModel):
    mode: str


class ActiveModelBody(BaseModel):
    model: str


class ThresholdBody(BaseModel):
    model_type: str
    scene: str
    threshold: float


class SensitivityBody(BaseModel):
    level: str
