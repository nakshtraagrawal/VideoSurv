from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/scores/{scene}/{video}")
def get_scores_json(scene: str, video: str):
    path = Path("outputs/scores") / f"{scene}_{video}_scores.npy"
    if not path.is_file():
        alt = list(Path("outputs/videos").glob(f"*{video}*_scores.npy"))
        path = alt[0] if alt else path
    if not path.is_file():
        raise HTTPException(404, "Scores not found")
    arr = np.load(str(path))
    return {"scores": arr.tolist()}
