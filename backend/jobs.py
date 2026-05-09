"""Simple in-memory job status for async inference/training (prototype)."""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"
    result: Dict[str, Any] = field(default_factory=dict)
    error: str | None = None


_LOCK = threading.Lock()
_JOBS: Dict[str, Job] = {}


def create_job(kind: str) -> Job:
    jid = str(uuid.uuid4())
    j = Job(id=jid, kind=kind)
    with _LOCK:
        _JOBS[jid] = j
    return j


def update_job(jid: str, **kwargs) -> None:
    with _LOCK:
        j = _JOBS.get(jid)
        if not j:
            return
        for k, v in kwargs.items():
            setattr(j, k, v)


def get_job(jid: str) -> Job | None:
    with _LOCK:
        return _JOBS.get(jid)
