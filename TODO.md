# AUTOSURVEIL — TODO

## Data & Environment
- [ ] Download Drone-Anomaly dataset (OneDrive link in repo README)
- [ ] Verify folder structure: `data/drone/{scene}/training|testing/frames/`
- [ ] `pip install -r requirements.txt`
- [ ] (Optional) Clone HSTforU + ASTNet reference repos for comparison — prototype uses self-contained `models/`

## Pipeline
- [x] `pipeline/preprocess.py` — video → frames extractor
- [x] `pipeline/dataset.py` — shared PyTorch Dataset for all 3 models
- [x] `pipeline/evaluate.py` — AUC-ROC, FPR, precision, recall
- [x] `pipeline/signal.py` — frame differencing + optical flow utils
- [x] `pipeline/infer.py` — run inference, produce output video

## Models
- [x] Conv Autoencoder — model.py, train.py, config.yaml
- [ ] Train Conv AE on bike scene (~20 min) → AUC > 0.65
- [x] ASTNet — model.py (attention residual AE), train.py, config.yaml
- [ ] Train ASTNet on bike scene (~1 hr) → AUC > 0.72
- [x] HSTforU — model.py, pvt_v2.py, hst_module.py, train.py, config.yaml
- [ ] Train HSTforU on bike scene (~2 hrs GPU) → AUC > 0.78
- [ ] Train all 3 models on all scenes (`scripts/train_all.sh`)
- [ ] Run `scripts/benchmark_all.sh` or POST `/api/benchmark/run` → export comparison JSON

## Backend
- [x] `database.py` — SQLite + SQLAlchemy setup
- [x] `main.py` — FastAPI entrypoint
- [x] `routers/system.py` — mode switching, scene list, stats, thresholds, active model
- [x] `routers/alerts.py` — alert CRUD
- [x] `routers/inference.py` — infer job + upload video
- [x] `routers/scores.py` — GET `/api/scores/{scene}/{video}`
- [x] `routers/training.py` — trigger training, get status
- [x] `routers/benchmark.py` — return benchmark results
- [x] `services/detector.py` — wraps model inference, keeps models loaded
- [x] `services/scorer.py` — anomaly scoring helpers
- [x] `services/feedback.py` — feedback ingestion, retraining trigger

## Frontend
- [x] Vite + React + Tailwind scaffold, `api/client.js`
- [x] `ModeSelector` — HARVEST / SURVEILLANCE / REVIEW toggle
- [x] `Dashboard.jsx` — live score feed, active alerts panel
- [x] `ScoreTimeline.jsx` — recharts anomaly score graph
- [x] `Alerts.jsx` — alert list + Confirm / Mark Normal buttons
- [x] `HeatmapVideo.jsx` — video player with heatmap overlay
- [x] `Benchmark.jsx` — model comparison table + AUC graphs
- [x] `Training.jsx` — training progress, harvest mode stats
- [x] `Settings.jsx` — threshold slider, scene config

## Output & Demo
- [ ] Generate heatmap overlay video for bike scene
- [ ] Hold-out test: train on 5 scenes, eval on bike + highway
- [ ] Week 8 demo: HARVEST → SURVEILLANCE → alert → feedback → retrain

## Repo layout (from PRD)
- [x] `scripts/` — download_data.sh, train_all.sh, benchmark_all.sh
- [x] `docker-compose.yml` + `Dockerfile.backend` + `frontend/Dockerfile`
- [x] `requirements.txt`
