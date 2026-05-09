# AUTOSURVEIL — Product Requirements Document
## V0-9-AUTOSURVEIL | Drone Surveillance Anomaly Detection | Complete Prototype

---

## 1. What we are building

A full-stack drone surveillance anomaly detection system. The operator flies a drone over an area, the system learns what normal looks like, and automatically flags anything unusual. Alerts go to a web dashboard where the operator reviews and provides feedback. The model improves over time from that feedback.

This is a research prototype. The goal is to prove the approach works on the Drone-Anomaly dataset, benchmark three different detection models side by side, and ship a working demo that a non-technical stakeholder can use.

**Not in scope for this prototype:** real-time video streaming from actual drones, multi-drone fusion, edge deployment on hardware.

---

## 2. System modes (user journey)

The system has three operational modes. This is the core UX concept.

### HARVEST MODE
The user designates this session as "data collection." Drone footage is recorded and stored as normal footage. No anomaly detection runs. The dashboard shows how many hours of normal data have been collected per scene and a confidence meter ("Model readiness: 34%") that increases as more normal data accumulates. This is how the system learns what normal looks like before being deployed.

For the prototype, the Drone-Anomaly training split fills this role. The dashboard shows it as "pre-harvested normal footage: 37 sequences loaded."

### SURVEILLANCE MODE
The trained model runs on incoming footage. Every clip is scored. When the anomaly score crosses the detection threshold, an alert fires. The dashboard shows a live feed of anomaly scores per scene, a notification panel for active alerts, and a video clip of the flagged footage with heatmap overlay.

### REVIEW MODE
The operator reviews fired alerts. Each alert shows: the flagged video clip with heatmap, the anomaly score, the timestamp, and the scene. Two actions: **Confirm Anomaly** (true positive, logged) or **Mark as Normal** (false positive, clip goes into next training round). The system tracks false positive rate over time. After enough new feedback accumulates, a retraining job is queued automatically.

---

## 3. Models — what to build and why

Three models are implemented on the same dataset so results can be directly compared. All three use the same data pipeline, same evaluation script, same metrics. The comparison table is shown on the dashboard.

### Model 1 — Conv Autoencoder (baseline)
Simple convolutional encoder-decoder. Train on normal frames only. Reconstruction error = anomaly score. Fast to train, easy to understand, weak on temporal/motion anomalies. This is the sanity check — everything else should beat it.

**Reference:** build from scratch (no external repo needed)
**Input:** single frame (256×256 grayscale)
**Anomaly score:** MSE(original, reconstruction)

### Model 2 — ASTNet (attention baseline)
Attention-based residual autoencoder. Adds spatial attention on top of the autoencoder — the model learns where to look. Better at localizing what's wrong within a frame. Middle ground between the simple AE and the full transformer.

**Reference:** https://github.com/vt-le/astnet (official implementation, MIT license)
**Input:** single frame (256×256)
**Anomaly score:** weighted reconstruction error using attention maps

### Model 3 — HSTforU (primary model)
Hierarchical Spatio-Temporal Transformer for U-Net. The best-performing model on the exact Drone-Anomaly dataset we are using. Uses a 4-stage pyramid transformer (PVTv2 backbone) as encoder. At each stage a spatio-temporal attention module adds temporal context across a clip. U-Net decoder with skip connections. Predicts next frame — normal events are predictable, anomalies are not.

**Reference:** https://github.com/vt-le/HSTforU (official implementation)
**Input:** 5 consecutive frames (256×256 grayscale)
**Anomaly score:** MSE(predicted next frame, actual next frame)
**Why this is the primary model:** it was specifically designed and benchmarked for aerial drone footage and outperforms all prior methods on Drone-Anomaly.

### Decision logic
Build all three. Use HSTforU for the live dashboard alerts. Show the comparison table on a benchmark page. This way the prototype is both a working product AND a research comparison.

---

## 4. Dataset

**Drone-Anomaly Dataset**
- Source: https://github.com/Jin-Pu/Drone-Anomaly (download from OneDrive link in README)
- 7 scenes: bike, highway, crossroads, and others
- 37 training sequences (normal footage only)
- 22 test sequences (mix of normal and anomalous)
- ~87,000 frames total, 640×640 at 30fps
- Annotations: per-frame binary labels as .npy files (0=normal, 1=anomaly)

**Dataset split for this prototype:**
- Training: all 37 provided training sequences (normal footage)
- Validation: 20% sampled from training sequences (held out during training)
- Test set A (in-distribution): all 22 provided test sequences
- Test set B (held-out scenes): hold out the `bike` and `highway` scenes entirely from training. Train only on remaining 5 scenes. Evaluate on bike and highway. This tests generalisation to unseen scenes — the realistic deployment scenario.

---

## 5. Project structure

```
autosurveil/
├── data/
│   └── drone/
│       ├── bike/
│       │   ├── training/frames/{video_name}/{frame}.jpg
│       │   ├── testing/frames/{video_name}/{frame}.jpg
│       │   └── annotation/{video_name}.npy
│       ├── highway/
│       ├── crossroads/
│       └── [other scenes]/
│
├── models/
│   ├── conv_autoencoder/
│   │   ├── model.py
│   │   ├── train.py
│   │   └── config.yaml
│   ├── astnet/
│   │   ├── model.py           # adapted from vt-le/astnet
│   │   ├── train.py
│   │   └── config.yaml
│   └── hstforu/
│       ├── model.py           # adapted from vt-le/HSTforU
│       ├── pvt_v2.py          # PVTv2 backbone
│       ├── hst_module.py      # Hierarchical Spatio-Temporal module
│       ├── train.py
│       └── config.yaml
│
├── pipeline/
│   ├── dataset.py             # shared PyTorch Dataset for all models
│   ├── preprocess.py          # video → frames extractor
│   ├── evaluate.py            # shared evaluation: AUC-ROC, FPR, precision, recall
│   ├── infer.py               # run inference on a video, produce output video
│   └── signal.py              # frame differencing + optical flow utilities
│
├── backend/
│   ├── main.py                # FastAPI app entrypoint
│   ├── routers/
│   │   ├── alerts.py          # alert CRUD endpoints
│   │   ├── training.py        # trigger training, get status
│   │   ├── inference.py       # run detection on uploaded video
│   │   ├── benchmark.py       # get benchmark results table
│   │   └── system.py          # system mode, scene list, stats
│   ├── models/
│   │   ├── database.py        # SQLite setup with SQLAlchemy
│   │   └── schemas.py         # Pydantic schemas
│   └── services/
│       ├── detector.py        # wraps model inference, keeps models loaded
│       ├── scorer.py          # anomaly scoring + thresholding
│       └── feedback.py        # feedback ingestion, retraining trigger
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx      # main surveillance view
│   │   │   ├── Alerts.jsx         # alert review + feedback
│   │   │   ├── Benchmark.jsx      # model comparison table + graphs
│   │   │   ├── Training.jsx       # training progress + harvest mode
│   │   │   └── Settings.jsx       # threshold, scene config
│   │   ├── components/
│   │   │   ├── ScoreTimeline.jsx  # anomaly score graph (recharts)
│   │   │   ├── AlertCard.jsx      # single alert with video + actions
│   │   │   ├── HeatmapVideo.jsx   # video player with heatmap overlay
│   │   │   ├── ModelBadge.jsx     # active model indicator
│   │   │   ├── ModeSelector.jsx   # HARVEST / SURVEILLANCE / REVIEW
│   │   │   └── MetricsTable.jsx   # AUC/FPR/precision/recall table
│   │   └── api/
│   │       └── client.js          # axios API client
│   ├── package.json
│   └── vite.config.js
│
├── checkpoints/               # saved model weights
├── outputs/
│   ├── videos/                # output videos with heatmap overlay
│   ├── scores/                # per-video anomaly score .npy files
│   └── benchmark/             # benchmark result JSONs
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_signal_extraction.ipynb
│   └── 03_model_comparison.ipynb
├── scripts/
│   ├── download_data.sh       # instructions + wget commands for dataset
│   ├── train_all.sh           # train all 3 models on all scenes
│   └── benchmark_all.sh       # evaluate all 3 models, output table
├── requirements.txt
├── README.md
└── docker-compose.yml         # optional: run backend + frontend together
```

---

## 6. Data pipeline (`pipeline/`)

### 6.1 `pipeline/preprocess.py`

Converts raw video files to frame folders matching the Drone-Anomaly structure. Also handles resizing and grayscale conversion.

```python
"""
preprocess.py
Usage:
    python pipeline/preprocess.py --input data/raw/ --output data/drone/ --size 256
"""
import cv2, os, argparse
from pathlib import Path

def extract_frames(video_path: str, output_dir: str, size: int = 256):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (size, size))
        cv2.imwrite(f"{output_dir}/{idx:04d}.jpg", frame)
        idx += 1
    cap.release()
    return idx
```

### 6.2 `pipeline/dataset.py`

Single PyTorch Dataset class used by all three models. Returns sliding window clips of N frames. Handles both single-frame mode (for Conv AE / ASTNet) and multi-frame clip mode (for HSTforU).

```python
"""
dataset.py
Shared dataset for all three models.
"""
import os, numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class DroneDataset(Dataset):
    """
    Args:
        scene_dir: path to scene (e.g. data/drone/bike)
        split: 'training' or 'testing'
        clip_length: frames per sample. Set 1 for AE/ASTNet, 5 for HSTforU
        img_size: resize to this square size
        stride: step between clips (1 = dense, clip_length = non-overlapping)
    """
    def __init__(self, scene_dir, split='training', clip_length=5,
                 img_size=256, stride=1):
        self.clip_length = clip_length
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.Grayscale(1),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])
        self.clips = []  # list of (video_path, [frame_files], start_idx)
        frames_root = os.path.join(scene_dir, split, 'frames')
        for vid in sorted(os.listdir(frames_root)):
            vid_path = os.path.join(frames_root, vid)
            if not os.path.isdir(vid_path): continue
            frames = sorted([f for f in os.listdir(vid_path) if f.endswith('.jpg')])
            for start in range(0, len(frames) - clip_length + 1, stride):
                self.clips.append((vid_path, frames, start))

        # Load annotations for test split
        self.annotations = {}
        if split == 'testing':
            ann_dir = os.path.join(scene_dir, 'annotation')
            if os.path.exists(ann_dir):
                for f in os.listdir(ann_dir):
                    if f.endswith('.npy'):
                        self.annotations[f[:-4]] = np.load(os.path.join(ann_dir, f))

    def __len__(self): return len(self.clips)

    def __getitem__(self, idx):
        vid_path, frames, start = self.clips[idx]
        clip = []
        for i in range(self.clip_length):
            img = Image.open(os.path.join(vid_path, frames[start + i])).convert('RGB')
            clip.append(self.transform(img))
        clip = torch.stack(clip, dim=0)   # (clip_length, C, H, W)
        return clip


class DroneTestDataset(DroneDataset):
    """Same as DroneDataset but also returns video name + target frame index."""
    def __getitem__(self, idx):
        vid_path, frames, start = self.clips[idx]
        clip = []
        for i in range(self.clip_length):
            img = Image.open(os.path.join(vid_path, frames[start + i])).convert('RGB')
            clip.append(self.transform(img))
        clip = torch.stack(clip, dim=0)
        vid_name = os.path.basename(vid_path)
        target_idx = start + self.clip_length - 1
        return clip, vid_name, target_idx
```

### 6.3 `pipeline/signal.py`

Classical signal extraction. Used in notebooks and for the signal extraction phase demo.

```python
"""
signal.py
Frame differencing, optical flow, camera motion compensation.
"""
import cv2
import numpy as np

def frame_diff(f1_gray, f2_gray, thresh=30):
    """Absolute frame difference → binary mask"""
    diff = cv2.absdiff(f1_gray, f2_gray)
    _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    return mask

def optical_flow_farneback(f1_gray, f2_gray):
    """Dense optical flow → (H,W,2) flow array"""
    return cv2.calcOpticalFlowFarneback(
        f1_gray, f2_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

def flow_to_hsv(frame_bgr, flow):
    """Visualise optical flow as colour-coded overlay"""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros_like(frame_bgr)
    hsv[..., 1] = 255
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def stabilise(f1_gray, f2_gray):
    """ORB + homography camera motion compensation. Returns warped f2."""
    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(f1_gray, None)
    kp2, des2 = orb.detectAndCompute(f2_gray, None)
    if des1 is None or des2 is None or len(kp1) < 4:
        return f2_gray
    matches = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, True).match(des1, des2),
                     key=lambda x: x.distance)[:50]
    if len(matches) < 4:
        return f2_gray
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    if H is None: return f2_gray
    h, w = f1_gray.shape
    return cv2.warpPerspective(f2_gray, H, (w, h))
```

### 6.4 `pipeline/evaluate.py`

Shared evaluation for all three models. Takes a scores array and ground truth labels, returns all metrics.

```python
"""
evaluate.py
Usage:
    python pipeline/evaluate.py \
        --model hstforu --scene data/drone/bike \
        --checkpoint checkpoints/hstforu_bike/best.pth
"""
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import matplotlib.pyplot as plt
import os, json

def compute_metrics(gt_labels: np.ndarray, scores: np.ndarray,
                    threshold_percentile: float = 95) -> dict:
    """
    Args:
        gt_labels: binary ground truth (0=normal, 1=anomaly)
        scores: raw anomaly scores (higher = more anomalous)
        threshold_percentile: percentile of scores to use as detection threshold
    Returns:
        dict with auc, precision, recall, fpr, threshold
    """
    # Normalise scores to [0,1]
    s_min, s_max = scores.min(), scores.max()
    norm = (scores - s_min) / (s_max - s_min + 1e-8)

    auc = roc_auc_score(gt_labels, norm)
    threshold = np.percentile(norm, threshold_percentile)
    binary_pred = (norm > threshold).astype(int)

    precision = precision_score(gt_labels, binary_pred, zero_division=0)
    recall = recall_score(gt_labels, binary_pred, zero_division=0)

    normal_mask = gt_labels == 0
    fpr = (np.sum((binary_pred == 1) & normal_mask) /
           (np.sum(normal_mask) + 1e-8))

    return {
        'auc': round(float(auc), 4),
        'precision': round(float(precision), 4),
        'recall': round(float(recall), 4),
        'fpr': round(float(fpr), 4),
        'threshold': round(float(threshold), 6)
    }

def smooth(scores: np.ndarray, window: int = 15) -> np.ndarray:
    """Rolling mean smoothing"""
    return np.convolve(scores, np.ones(window)/window, mode='same')

def plot_score_timeline(scores, gt_labels, video_name, model_name,
                        output_path):
    """Save anomaly score timeline graph with GT overlay"""
    fig, ax = plt.subplots(figsize=(14, 4))
    frames = np.arange(len(scores))
    ax.plot(frames, scores, color='steelblue', lw=1, label='Anomaly Score')
    thresh = np.percentile(scores, 95)
    ax.axhline(thresh, color='orange', lw=1, ls='--', label='Threshold')
    # Shade GT anomaly regions
    in_anom, start = False, 0
    for i, lbl in enumerate(gt_labels):
        if lbl == 1 and not in_anom:
            start, in_anom = i, True
        elif lbl == 0 and in_anom:
            ax.axvspan(start, i, alpha=0.25, color='red', label='GT Anomaly')
            in_anom = False
    if in_anom:
        ax.axvspan(start, len(gt_labels), alpha=0.25, color='red')
    ax.set(xlabel='Frame', ylabel='Score',
           title=f'{model_name} | {video_name}')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys())
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
```

### 6.5 `pipeline/infer.py`

Runs any of the three models on a folder of frames, produces an output .mp4 with heatmap overlay.

```python
"""
infer.py
Usage:
    python pipeline/infer.py \
        --frames data/drone/bike/testing/frames/01 \
        --model hstforu \
        --checkpoint checkpoints/hstforu_bike/best.pth \
        --output outputs/videos/bike_01.mp4 \
        --threshold 0.015
"""
import argparse, os, cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from collections import deque


def load_model(model_type: str, checkpoint: str, device, clip_length=5, img_size=256):
    if model_type == 'conv_ae':
        from models.conv_autoencoder.model import ConvAutoencoder
        m = ConvAutoencoder()
        m.load_state_dict(torch.load(checkpoint, map_location=device))
    elif model_type == 'astnet':
        from models.astnet.model import ASTNet
        m = ASTNet()
        m.load_state_dict(torch.load(checkpoint, map_location=device))
    elif model_type == 'hstforu':
        from models.hstforu.model import HSTforU
        m = HSTforU(clip_length=clip_length, img_size=img_size, in_channels=1)
        m.load_state_dict(torch.load(checkpoint, map_location=device))
    return m.to(device).eval()


def make_heatmap_overlay(frame_bgr, score_map_8x8, alpha=0.4):
    h, w = frame_bgr.shape[:2]
    sm = cv2.resize(score_map_8x8.astype(np.float32), (w, h),
                    interpolation=cv2.INTER_LINEAR)
    sm = cv2.GaussianBlur(sm, (21, 21), 0)
    if sm.max() > sm.min():
        sm = ((sm - sm.min()) / (sm.max() - sm.min()) * 255).astype(np.uint8)
    else:
        sm = np.zeros((h, w), dtype=np.uint8)
    heatmap = cv2.applyColorMap(sm, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame_bgr, 1 - alpha, heatmap, alpha, 0)


def add_ui_overlay(frame, score, threshold, frame_idx, is_anomaly):
    h, w = frame.shape[:2]
    # Score bar
    bar_y = h - 28
    cv2.rectangle(frame, (10, bar_y), (w-10, bar_y+18), (40,40,40), -1)
    fill = int(min(score, threshold*2) / (threshold*2) * (w-20))
    color = (0,60,255) if is_anomaly else (0,200,80)
    cv2.rectangle(frame, (10, bar_y), (10+fill, bar_y+18), color, -1)
    cv2.putText(frame, f'SCORE: {score:.5f}  THR: {threshold:.5f}',
                (14, bar_y+13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1)
    # Alert banner
    if is_anomaly:
        cv2.rectangle(frame, (0,0), (w, 36), (0,0,180), -1)
        cv2.putText(frame, 'ANOMALY DETECTED', (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    # Frame counter
    cv2.putText(frame, f'#{frame_idx:05d}', (w-80, h-35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180,180,180), 1)
    return frame


def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model, args.checkpoint, device,
                       clip_length=5, img_size=256)

    transform = T.Compose([
        T.Resize((256,256)), T.Grayscale(1),
        T.ToTensor(), T.Normalize([0.5],[0.5])
    ])

    frames_files = sorted([f for f in os.listdir(args.frames) if f.endswith('.jpg')])
    sample_frame = cv2.imread(os.path.join(args.frames, frames_files[0]))
    H, W = sample_frame.shape[:2]

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    writer = cv2.VideoWriter(args.output,
                             cv2.VideoWriter_fourcc(*'mp4v'), 30, (W, H))

    clip_buf = deque(maxlen=5)
    score_buf = deque(maxlen=15)
    all_scores = []

    for i, fname in enumerate(frames_files):
        bgr = cv2.imread(os.path.join(args.frames, fname))
        t = transform(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        clip_buf.append(t)

        score = 0.0
        score_map = np.zeros((8,8))

        if len(clip_buf) == 5:
            with torch.no_grad():
                if args.model == 'conv_ae':
                    x = t.unsqueeze(0).to(device)
                    recon = model(x)
                    score = float(torch.mean((x - recon)**2).item())
                    # Patch-level scores for heatmap
                    err = (x - recon).abs().squeeze().cpu().numpy()
                    score_map = cv2.resize(err, (8,8))
                elif args.model == 'astnet':
                    x = t.unsqueeze(0).to(device)
                    recon, _ = model(x)
                    score = float(torch.mean((x - recon)**2).item())
                    err = (x - recon).abs().squeeze().cpu().numpy()
                    score_map = cv2.resize(err, (8,8))
                elif args.model == 'hstforu':
                    clip = torch.stack(list(clip_buf)).unsqueeze(0).to(device)
                    pred = model(clip[:, :-1])
                    target = clip[:, -1]
                    err_map = torch.mean((pred - target)**2, dim=1).squeeze().cpu().numpy()
                    score = float(err_map.mean())
                    score_map = cv2.resize(err_map, (8,8))

        score_buf.append(score)
        smoothed = float(np.mean(score_buf))
        all_scores.append(smoothed)

        out = make_heatmap_overlay(bgr.copy(), score_map)
        out = add_ui_overlay(out, smoothed, args.threshold, i,
                             smoothed > args.threshold)
        writer.write(out)

        if i % 200 == 0:
            print(f"  {i}/{len(frames_files)}  score={smoothed:.5f}")

    writer.release()
    # Save scores
    scores_path = args.output.replace('.mp4', '_scores.npy')
    np.save(scores_path, np.array(all_scores))
    print(f"Done. Video: {args.output}  Scores: {scores_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--frames', required=True)
    p.add_argument('--model', required=True, choices=['conv_ae','astnet','hstforu'])
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', default='outputs/videos/result.mp4')
    p.add_argument('--threshold', type=float, default=0.015)
    run(p.parse_args())
```

---

## 7. Models (`models/`)

### 7.1 Conv Autoencoder (`models/conv_autoencoder/model.py`)

```python
"""
Conv Autoencoder — baseline anomaly detector.
Train on normal frames only. High reconstruction error = anomaly.
"""
import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self, in_ch=1, latent=512):
        super().__init__()
        def conv_block(ic, oc, stride=2):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, stride, 1),
                nn.BatchNorm2d(oc), nn.LeakyReLU(0.2))
        def deconv_block(ic, oc):
            return nn.Sequential(
                nn.ConvTranspose2d(ic, oc, 4, 2, 1),
                nn.BatchNorm2d(oc), nn.ReLU())

        self.enc = nn.Sequential(
            conv_block(in_ch, 32),   # 128
            conv_block(32, 64),      # 64
            conv_block(64, 128),     # 32
            conv_block(128, 256),    # 16
        )
        self.fc_enc = nn.Sequential(nn.Flatten(),
                                    nn.Linear(256*16*16, latent))
        self.fc_dec = nn.Sequential(nn.Linear(latent, 256*16*16),
                                    nn.ReLU())
        self.dec = nn.Sequential(
            deconv_block(256, 128),
            deconv_block(128, 64),
            deconv_block(64, 32),
            nn.ConvTranspose2d(32, in_ch, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.fc_enc(self.enc(x))
        x_hat = self.dec(self.fc_dec(z).view(-1, 256, 16, 16))
        return x_hat

    def anomaly_score(self, x):
        with torch.no_grad():
            return torch.mean((x - self(x))**2, dim=[1,2,3])
```

**`models/conv_autoencoder/train.py`**

```python
"""
Usage: python models/conv_autoencoder/train.py --scene data/drone/bike --epochs 50
"""
import argparse, os, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from pipeline.dataset import DroneDataset
from models.conv_autoencoder.model import ConvAutoencoder

def train(args):
    wandb.init(project='autosurveil', name=f'conv_ae_{os.path.basename(args.scene)}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = DroneDataset(args.scene, 'training', clip_length=1, img_size=256, stride=1)
    loader = DataLoader(ds, args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = ConvAutoencoder().to(device)
    opt = Adam(model.parameters(), lr=args.lr)
    sched = CosineAnnealingLR(opt, args.epochs)
    crit = nn.MSELoss()
    best = float('inf')

    for epoch in range(args.epochs):
        model.train()
        total = 0
        for batch in loader:
            x = batch.squeeze(1).to(device)   # (B, C, H, W)
            loss = crit(model(x), x)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        avg = total / len(loader)
        sched.step()
        wandb.log({'train_loss': avg, 'epoch': epoch})
        print(f'[{epoch+1}/{args.epochs}] loss={avg:.6f}')
        if avg < best:
            best = avg
            os.makedirs(f'checkpoints/conv_ae_{os.path.basename(args.scene)}', exist_ok=True)
            torch.save(model.state_dict(),
                       f'checkpoints/conv_ae_{os.path.basename(args.scene)}/best.pth')
    wandb.finish()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    train(p.parse_args())
```

**`models/conv_autoencoder/config.yaml`**
```yaml
model: conv_autoencoder
in_channels: 1
latent_dim: 512
img_size: 256
clip_length: 1
train:
  epochs: 50
  batch_size: 32
  lr: 0.001
  num_workers: 4
eval:
  threshold_percentile: 95
  smooth_window: 15
```

---

### 7.2 ASTNet (`models/astnet/`)

ASTNet is an attention-based residual autoencoder. Clone the official repo and adapt it.

```bash
git clone https://github.com/vt-le/astnet.git astnet_reference
```

**`models/astnet/model.py`** — adapt from `astnet_reference/ASTNet/`

ASTNet architecture summary:
- Wide ResNet feature extractor as encoder backbone
- Spatial self-attention modules on encoder feature maps
- Residual decoder reconstructs the input frame
- Anomaly score = weighted reconstruction error (attention-guided)

```python
"""
models/astnet/model.py
Adapted from https://github.com/vt-le/astnet (MIT License)
Minimal changes: made clip_length and img_size configurable,
removed dataset-specific assumptions.
"""
# Copy ASTNet/models/ from astnet_reference here.
# Main class is ASTNet(cfg) where cfg is loaded from config.yaml.
# For our use:
#   forward(x) returns (reconstructed_frame, attention_maps)
#   anomaly_score(x) returns MSE weighted by attention maps
#
# See astnet_reference/ASTNet/models/ for full implementation.
# Key files to copy: model.py, loss.py, networks/ folder
```

**`models/astnet/config.yaml`**
```yaml
model: astnet
backbone: wide_resnet
in_channels: 1
img_size: 256
clip_length: 1
train:
  epochs: 60
  batch_size: 16
  lr: 0.0002
  num_workers: 4
eval:
  threshold_percentile: 95
  smooth_window: 15
```

**`models/astnet/train.py`** — same structure as conv_autoencoder/train.py, adapted for ASTNet's forward signature:
```python
# forward returns (recon, attn_maps)
# loss = MSELoss(recon, x)
# Optionally add attention regularisation loss from original paper
```

---

### 7.3 HSTforU (`models/hstforu/`)

```bash
git clone https://github.com/vt-le/HSTforU.git hstforu_reference
```

HSTforU architecture summary:
- Input: N=5 consecutive frames → predict frame N+1
- Encoder: PVTv2-B2 backbone, 4 stages producing feature maps at 1/4, 1/8, 1/16, 1/32 resolution
- At each stage: HST (Hierarchical Spatio-Temporal) module adds temporal attention across the clip
- Decoder: U-Net style, upsamples with skip connections from each encoder stage
- Output: predicted next frame
- Anomaly score: MSE(predicted, actual)

**`models/hstforu/model.py`**
```python
"""
models/hstforu/model.py
Adapted from https://github.com/vt-le/HSTforU (official implementation)
Copy HSTforU/models/ from hstforu_reference.
Key class: HSTforU(encoder_name, clip_length, img_size, in_channels)
forward(x) where x is shape (B, clip_length-1, C, H, W)
returns predicted next frame (B, C, H, W)
"""
import sys
sys.path.insert(0, 'hstforu_reference')
from HSTforU.models.model import HSTforU  # re-export
```

**`models/hstforu/config.yaml`**
```yaml
model: hstforu
encoder: pvt_v2_b2
in_channels: 1
img_size: 256
clip_length: 5          # 4 input frames → predict 5th
train:
  epochs: 100
  batch_size: 8          # reduce to 4 if GPU OOM
  lr: 0.0001
  weight_decay: 0.0001
  warmup_epochs: 10
  num_workers: 4
eval:
  threshold_percentile: 95
  smooth_window: 15
```

**`models/hstforu/train.py`**
```python
"""
Usage: python models/hstforu/train.py --scene data/drone/bike
"""
import argparse, os, torch, sys
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from pipeline.dataset import DroneDataset

sys.path.insert(0, 'hstforu_reference')
from HSTforU.models.model import HSTforU

def train(args):
    wandb.init(project='autosurveil', name=f'hstforu_{os.path.basename(args.scene)}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = DroneDataset(args.scene, 'training', clip_length=5, img_size=256, stride=1)
    loader = DataLoader(ds, args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    model = HSTforU(encoder_name='pvt_v2_b2', clip_length=5,
                    img_size=256, in_channels=1).to(device)
    print(f'Params: {sum(p.numel() for p in model.parameters()):,}')

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, args.epochs)
    crit = nn.MSELoss()
    best = float('inf')

    for epoch in range(args.epochs):
        model.train()
        total = 0
        for clips in loader:
            # clips: (B, 5, C, H, W)
            clips = clips.to(device)
            inp = clips[:, :-1]   # 4 frames in
            tgt = clips[:, -1]    # predict 5th
            pred = model(inp)
            loss = crit(pred, tgt)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        avg = total / len(loader)
        sched.step()
        wandb.log({'train_loss': avg, 'lr': sched.get_last_lr()[0], 'epoch': epoch})
        print(f'[{epoch+1}/{args.epochs}] loss={avg:.6f}')
        if avg < best:
            best = avg
            os.makedirs(f'checkpoints/hstforu_{os.path.basename(args.scene)}', exist_ok=True)
            torch.save(model.state_dict(),
                       f'checkpoints/hstforu_{os.path.basename(args.scene)}/best.pth')
    wandb.finish()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    train(p.parse_args())
```

---

## 8. Backend (`backend/`)

FastAPI. All endpoints are REST. No WebSockets for prototype.

### 8.1 `backend/models/database.py`

```python
"""SQLite database with SQLAlchemy"""
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

DATABASE_URL = "sqlite:///./autosurveil.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Alert(Base):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True, index=True)
    scene = Column(String)
    video_name = Column(String)
    frame_idx = Column(Integer)
    anomaly_score = Column(Float)
    model_used = Column(String)
    clip_path = Column(String)         # path to saved .mp4 clip
    heatmap_path = Column(String)      # path to heatmap image
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    reviewed = Column(Boolean, default=False)
    confirmed_anomaly = Column(Boolean, nullable=True)  # None=unreviewed
    feedback_note = Column(String, nullable=True)

class TrainingRun(Base):
    __tablename__ = "training_runs"
    id = Column(Integer, primary_key=True, index=True)
    model_type = Column(String)
    scene = Column(String)
    status = Column(String)            # queued | running | complete | failed
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    final_loss = Column(Float, nullable=True)
    auc = Column(Float, nullable=True)
    checkpoint_path = Column(String, nullable=True)

class BenchmarkResult(Base):
    __tablename__ = "benchmark_results"
    id = Column(Integer, primary_key=True, index=True)
    model_type = Column(String)
    scene = Column(String)
    auc = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    fpr = Column(Float)
    threshold = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(bind=engine)
```

### 8.2 `backend/models/schemas.py`

```python
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class AlertOut(BaseModel):
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
    class Config: orm_mode = True

class AlertFeedback(BaseModel):
    confirmed_anomaly: bool
    feedback_note: Optional[str] = None

class TrainingRequest(BaseModel):
    model_type: str  # conv_ae | astnet | hstforu
    scene: str
    epochs: Optional[int] = None  # uses config default if None

class TrainingRunOut(BaseModel):
    id: int
    model_type: str
    scene: str
    status: str
    final_loss: Optional[float]
    auc: Optional[float]
    class Config: orm_mode = True

class BenchmarkResultOut(BaseModel):
    model_type: str
    scene: str
    auc: float
    precision: float
    recall: float
    fpr: float
    class Config: orm_mode = True

class InferRequest(BaseModel):
    frames_dir: str   # path to frames folder
    model_type: str
    scene: str        # to select correct checkpoint
    threshold: Optional[float] = None  # uses saved threshold if None

class SystemStatus(BaseModel):
    mode: str         # harvest | surveillance | review
    active_model: str
    scenes_available: list
    total_alerts: int
    unreviewed_alerts: int
    harvest_hours: float
```

### 8.3 API endpoints

**`backend/routers/system.py`**
```
GET  /api/system/status           → SystemStatus
POST /api/system/mode             → set mode (harvest|surveillance|review)
GET  /api/system/scenes           → list available scenes from data/drone/
```

**`backend/routers/alerts.py`**
```
GET  /api/alerts                  → list alerts (filter: scene, reviewed, confirmed)
GET  /api/alerts/{id}             → single alert detail
POST /api/alerts/{id}/feedback    → AlertFeedback → mark reviewed + label
GET  /api/alerts/{id}/clip        → serve video clip file
GET  /api/alerts/{id}/heatmap     → serve heatmap image
DELETE /api/alerts/{id}           → delete alert
```

**`backend/routers/inference.py`**
```
POST /api/infer                   → InferRequest → kicks off inference job
                                    returns job_id
GET  /api/infer/{job_id}          → job status + results when done
POST /api/infer/upload            → upload a video file, run inference
GET  /api/scores/{scene}/{video}  → get anomaly score array as JSON
```

**`backend/routers/training.py`**
```
POST /api/training/start          → TrainingRequest → queues training job
GET  /api/training/{id}           → TrainingRunOut (with status)
GET  /api/training/history        → all past training runs
POST /api/training/retrain        → trigger retraining on feedback data
                                    (uses all operator-confirmed normal clips)
```

**`backend/routers/benchmark.py`**
```
GET  /api/benchmark               → all BenchmarkResults
POST /api/benchmark/run           → run evaluate.py on all 3 models, save results
GET  /api/benchmark/summary       → grouped by model, best AUC per scene
```

### 8.4 `backend/services/detector.py`

Loads all three models at startup and keeps them in memory. Provides a unified `score(frames, model_type)` interface.

```python
"""
detector.py
Singleton that loads all three models once at startup.
"""
import torch
import numpy as np
from collections import defaultdict

class Detector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}           # model_type -> loaded model
        self.thresholds = {}       # (model_type, scene) -> float
        self.active_model = 'hstforu'

    def load(self, model_type: str, checkpoint_path: str, scene: str):
        """Load a checkpoint into memory"""
        from pipeline.infer import load_model
        self.models[f'{model_type}_{scene}'] = load_model(
            model_type, checkpoint_path, self.device
        )
        print(f'Loaded {model_type} for {scene}')

    def score_clip(self, frames_tensors, model_type: str, scene: str) -> float:
        """Score a single clip. Returns anomaly score float."""
        key = f'{model_type}_{scene}'
        if key not in self.models:
            raise ValueError(f'Model {key} not loaded')
        model = self.models[key]
        # ... inference logic (same as infer.py)
        pass

    def set_threshold(self, model_type: str, scene: str, threshold: float):
        self.thresholds[(model_type, scene)] = threshold

    def get_threshold(self, model_type: str, scene: str) -> float:
        return self.thresholds.get((model_type, scene), 0.015)

detector = Detector()  # module-level singleton
```

### 8.5 `backend/main.py`

```python
"""FastAPI entrypoint"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.routers import alerts, training, inference, benchmark, system

app = FastAPI(title='AUTOSURVEIL API', version='0.1.0')

app.add_middleware(CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://localhost:3000'],
    allow_methods=['*'], allow_headers=['*'])

app.include_router(system.router,    prefix='/api/system')
app.include_router(alerts.router,    prefix='/api/alerts')
app.include_router(inference.router, prefix='/api/infer')
app.include_router(training.router,  prefix='/api/training')
app.include_router(benchmark.router, prefix='/api/benchmark')

# Serve output files (videos, heatmaps)
app.mount('/outputs', StaticFiles(directory='outputs'), name='outputs')

@app.on_event('startup')
async def startup():
    # Auto-load best checkpoints on startup
    import os, glob
    for ckpt in glob.glob('checkpoints/*/best.pth'):
        parts = os.path.basename(os.path.dirname(ckpt)).split('_', 1)
        if len(parts) == 2:
            model_type, scene = parts[0], parts[1]
            try:
                from backend.services.detector import detector
                detector.load(model_type, ckpt, scene)
            except Exception as e:
                print(f'Failed to load {ckpt}: {e}')
```

Run: `uvicorn backend.main:app --reload --port 8000`

---

## 9. Frontend (`frontend/`)

React + Vite + Tailwind. Dark theme. Military-inspired aesthetic. Minimal dependencies.

### 9.1 Tech stack
- React 18
- Vite
- Tailwind CSS
- Recharts (anomaly score timeline graphs)
- Axios (API calls)
- React Router (page routing)

### 9.2 Pages

#### `Dashboard.jsx` — Main surveillance view
```
Layout: sidebar nav + main content area

Sidebar:
  - AUTOSURVEIL logo
  - Mode indicator badge (HARVEST / SURVEILLANCE / REVIEW)
  - Nav: Dashboard, Alerts, Benchmark, Training, Settings
  - Active model badge (Conv AE / ASTNet / HSTforU)
  - System stats: total alerts, unreviewed count

Main content:
  - Scene selector tabs (one per scene: bike, highway, crossroads...)
  - Per scene: anomaly score timeline (last 1000 frames, live graph)
  - Recent alerts panel (last 5, each shows thumbnail + score + time)
  - Model confidence bar (based on hours of training data)
```

#### `Alerts.jsx` — Alert review and feedback
```
Layout: list view on left, detail panel on right

Left panel:
  - Filter bar: All / Unreviewed / Confirmed / False Positive
  - Scene filter dropdown
  - Sorted by time descending
  - Each row: thumbnail, scene, score, time, reviewed badge

Detail panel (when alert selected):
  - Video player (clip with heatmap overlay)
  - Anomaly score: X.XXXXX
  - Scene: bike, Frame: 00234
  - Time: 14:32:04
  - Two buttons: [✓ Confirm Anomaly] [✗ Mark as Normal]
  - Notes textarea
  - Submit feedback button
```

#### `Benchmark.jsx` — Model comparison
```
Top: Run Benchmark button (calls POST /api/benchmark/run)
Status indicator while running

Results table:
  Columns: Model | Scene | AUC-ROC | Precision | Recall | FPR | Status
  Color coding: green=best per column, red=worst
  
Per-scene AUC bar chart (recharts BarChart)
  X: Scene names
  Y: AUC value
  Three bars per scene (one per model), grouped

Best model recommendation box:
  "For bike scene, HSTforU achieves highest AUC (0.XX)"
```

#### `Training.jsx` — Training management and harvest mode
```
Harvest status panel:
  - Per scene: frames collected, hours of footage
  - Model readiness gauge (0-100%)
  - "Start Harvest Session" button (just records that user is in harvest mode)

Training jobs panel:
  - Start new training job: select model + scene + epochs
  - Active job: progress bar (polls GET /api/training/{id} every 2s)
  - Training history table: model, scene, AUC achieved, date

Retrain from feedback button:
  - Shows count of operator-labelled normal clips
  - "Queue retraining with X new normal samples"
```

#### `Settings.jsx`
```
Active model selector: Conv AE / ASTNet / HSTforU
Per-scene threshold sliders (one per scene)
Alert sensitivity: Low / Medium / High (maps to threshold percentile)
Smooth window size (frames)
```

### 9.3 Key components

**`ScoreTimeline.jsx`**
```jsx
// Recharts LineChart showing anomaly score over last N frames
// Red dashed horizontal line at current threshold
// Red shaded regions where score > threshold
// Props: scores (array), threshold (float), label (string)
```

**`AlertCard.jsx`**
```jsx
// Single alert card for the list view
// Props: alert (AlertOut schema)
// Shows: thumbnail (first frame of clip), scene badge, score, time, reviewed status
// Clickable → selects this alert in detail panel
```

**`HeatmapVideo.jsx`**
```jsx
// Video player for alert clips
// Uses HTML5 <video> tag
// Clip source: /outputs/videos/{clip_path}
// Plays automatically, loops
// Overlay: anomaly score badge in corner
```

**`ModelBadge.jsx`**
```jsx
// Shows active model name and its benchmark AUC
// Color coded: blue=conv_ae, yellow=astnet, green=hstforu
// Clicking opens Settings
```

**`ModeSelector.jsx`**
```jsx
// Three-state toggle: HARVEST | SURVEILLANCE | REVIEW
// Each state has a colour: grey=harvest, red=surveillance, blue=review
// Calls POST /api/system/mode on click
// Shows confirmation modal before switching to SURVEILLANCE
```

### 9.4 `frontend/src/api/client.js`

```javascript
import axios from 'axios'

const API = axios.create({ baseURL: 'http://localhost:8000' })

export const getStatus = () => API.get('/api/system/status')
export const setMode = (mode) => API.post('/api/system/mode', { mode })
export const getScenes = () => API.get('/api/system/scenes')

export const getAlerts = (params) => API.get('/api/alerts', { params })
export const getAlert = (id) => API.get(`/api/alerts/${id}`)
export const submitFeedback = (id, data) => API.post(`/api/alerts/${id}/feedback`, data)

export const startTraining = (data) => API.post('/api/training/start', data)
export const getTrainingRun = (id) => API.get(`/api/training/${id}`)
export const getTrainingHistory = () => API.get('/api/training/history')

export const runBenchmark = () => API.post('/api/benchmark/run')
export const getBenchmarkResults = () => API.get('/api/benchmark')

export const runInference = (data) => API.post('/api/infer', data)
export const getInferJob = (id) => API.get(`/api/infer/${id}`)
```

---

## 10. Environment and dependencies

### `requirements.txt`
```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
timm>=0.9.0
einops>=0.7.0
yacs>=0.1.8
pyyaml>=6.0
tqdm>=4.65.0
fastapi>=0.104.0
uvicorn>=0.24.0
sqlalchemy>=2.0.0
pydantic>=2.0.0
python-multipart>=0.0.6
aiofiles>=23.0.0
wandb>=0.16.0
scipy>=1.11.0
pandas>=2.0.0
Pillow>=10.0.0
```

### `frontend/package.json` (key deps)
```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.18.0",
    "recharts": "^2.9.0",
    "axios": "^1.5.0"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.1.0",
    "vite": "^4.5.0",
    "tailwindcss": "^3.3.0",
    "autoprefixer": "^10.4.16",
    "postcss": "^8.4.31"
  }
}
```

---

## 11. Setup and run (complete sequence)

```bash
# 1. Clone reference implementations
git clone https://github.com/vt-le/HSTforU.git hstforu_reference
git clone https://github.com/vt-le/astnet.git astnet_reference

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Download Drone-Anomaly dataset
# Go to https://github.com/Jin-Pu/Drone-Anomaly
# Download from OneDrive, extract to data/drone/
# Verify structure: data/drone/bike/training/frames/...

# 4. Install and run frontend
cd frontend
npm install
npm run dev    # runs at http://localhost:5173

# 5. Run backend (separate terminal)
cd ..
uvicorn backend.main:app --reload --port 8000

# 6. Train Conv Autoencoder on bike scene (fastest, ~20 min)
python models/conv_autoencoder/train.py --scene data/drone/bike --epochs 50

# 7. Train HSTforU on bike scene (~2 hrs with GPU)
python models/hstforu/train.py --scene data/drone/bike --epochs 100

# 8. Train ASTNet on bike scene (~1 hr)
python models/astnet/train.py --scene data/drone/bike --epochs 60

# 9. Run benchmark on all 3 models
python pipeline/evaluate.py --model conv_ae \
    --scene data/drone/bike \
    --checkpoint checkpoints/conv_ae_bike/best.pth

python pipeline/evaluate.py --model astnet \
    --scene data/drone/bike \
    --checkpoint checkpoints/astnet_bike/best.pth

python pipeline/evaluate.py --model hstforu \
    --scene data/drone/bike \
    --checkpoint checkpoints/hstforu_bike/best.pth

# 10. Generate output video
python pipeline/infer.py \
    --frames data/drone/bike/testing/frames/01 \
    --model hstforu \
    --checkpoint checkpoints/hstforu_bike/best.pth \
    --output outputs/videos/bike_01_hstforu.mp4 \
    --threshold 0.015

# 11. Train on all scenes
for scene in bike highway crossroads; do
    python models/hstforu/train.py --scene data/drone/$scene --epochs 100
done
```

---

## 12. Expected results

| Model | AUC-ROC (target) | Notes |
|---|---|---|
| Conv Autoencoder | > 0.65 | Baseline. Should work but misses temporal anomalies |
| ASTNet | > 0.72 | Better spatial localisation via attention |
| HSTforU | > 0.78 | Best: temporal + spatial, designed for drone footage |

False positive rate target for all models: < 10% on normal-only footage.

---

## 13. What to show in weekly meets

| Week | What to demo |
|---|---|
| 1 | Dashboard live, signal extraction notebook running, motion mask video |
| 2 | Backend API running, all 3 models training (show W&B loss curves) |
| 3 | Conv AE done: reconstruction visualisation + first AUC number |
| 4 | ASTNet done: comparison table (AE vs ASTNet), Alerts page working |
| 5 | HSTforU done: best AUC achieved, score timeline graph |
| 6 | Output video with heatmap overlay, Benchmark page showing all results |
| 7 | Full pipeline: upload video → inference → alert appears → operator feedback |
| 8 | Complete demo: HARVEST → SURVEILLANCE → alert → feedback → retrain queued |

---

## 14. Known limitations — tell stakeholders explicitly

- Trained per-scene: the model for `bike` only works for `bike`. Cross-scene generalisation is a Phase 2 problem.
- No live streaming: prototype processes pre-recorded video files only.
- Camera stabilisation is approximate: homography-based, good enough for prototype, not robust to fast drone movement.
- SQLite: fine for prototype, needs Postgres for multi-user production.
- No authentication: prototype assumes single trusted operator.
- No real-time optimisation: HSTforU at ~5-10 fps on a consumer GPU. Needs ONNX export for faster inference.

---

## 15. Phase 2 scope (write this for the next meeting)

1. Collect real drone footage for baseline (harvest mode, real flights)
2. Fine-tune HSTforU on real aerial data
3. Adaptive baseline: scheduled nightly retraining on operator-confirmed normal clips
4. Multi-scene generalisation: train on all scenes jointly
5. Multi-drone fusion: N parallel feeds → alert correlation layer
6. Real-time streaming: RTSP input → frame buffer → live inference
7. ONNX export for edge deployment

---

*V0-9-AUTOSURVEIL | Models: ConvAE + ASTNet (vt-le/astnet, MIT) + HSTforU (vt-le/HSTforU, published 2025) | Dataset: Drone-Anomaly (Jin-Pu, 2022)*
