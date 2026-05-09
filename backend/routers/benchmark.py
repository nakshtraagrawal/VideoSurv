from __future__ import annotations

import json
import threading
from collections import defaultdict
from pathlib import Path

import torch
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.models.database import BenchmarkResult, SessionLocal, get_db
from backend.models.schemas import BenchmarkResultOut
from pipeline.evaluate import compute_metrics, scores_for_scene

router = APIRouter()


@router.get("", response_model=list[BenchmarkResultOut])
def all_results(db: Session = Depends(get_db)):
    rows = db.query(BenchmarkResult).order_by(BenchmarkResult.id.desc()).limit(500).all()
    return [
        BenchmarkResultOut(
            model_type=r.model_type,
            scene=r.scene,
            auc=r.auc,
            precision=r.precision,
            recall=r.recall,
            fpr=r.fpr,
            threshold=r.threshold,
        )
        for r in rows
    ]


@router.get("/summary")
def summary(db: Session = Depends(get_db)):
    rows = db.query(BenchmarkResult).all()
    by_scene: dict[str, dict] = defaultdict(dict)
    for r in rows:
        by_scene[r.scene][r.model_type] = r.auc
    best_lines = []
    for scene, m in by_scene.items():
        if not m:
            continue
        best_model = max(m, key=lambda k: m[k])
        best_lines.append(
            {
                "scene": scene,
                "best_model": best_model,
                "auc": m[best_model],
            }
        )
    return {"per_scene_best": best_lines, "raw": {k: dict(v) for k, v in by_scene.items()}}


@router.post("/run")
def run_benchmark(db: Session = Depends(get_db)):
    def _job():
        db2 = SessionLocal()
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            scenes = [
                p.name
                for p in Path("data/drone").iterdir()
                if p.is_dir() and (p / "testing" / "frames").is_dir()
            ]
            models = ["conv_ae", "astnet", "hstforu"]
            out_dir = Path("outputs/benchmark")
            out_dir.mkdir(parents=True, exist_ok=True)
            summary_rows = []
            for scene in scenes:
                scene_path = str(Path("data/drone") / scene)
                for mt in models:
                    ckpt = Path("checkpoints") / f"{mt}_{scene}" / "best.pth"
                    if not ckpt.is_file():
                        continue
                    gt, sc, _ = scores_for_scene(
                        mt, scene_path, str(ckpt), device=device
                    )
                    if len(gt) == 0:
                        continue
                    metrics = compute_metrics(gt, sc)
                    row = BenchmarkResult(
                        model_type=mt,
                        scene=scene,
                        auc=metrics["auc"],
                        precision=metrics["precision"],
                        recall=metrics["recall"],
                        fpr=metrics["fpr"],
                        threshold=metrics["threshold"],
                    )
                    db2.add(row)
                    summary_rows.append({"model": mt, "scene": scene, **metrics})
                    with open(
                        out_dir / f"{mt}_{scene}.json", "w", encoding="utf-8"
                    ) as f:
                        json.dump(metrics, f, indent=2)
            db2.commit()
            with open(out_dir / "latest_table.json", "w", encoding="utf-8") as f:
                json.dump(summary_rows, f, indent=2)
        finally:
            db2.close()

    threading.Thread(target=_job, daemon=True).start()
    return {"status": "started"}
