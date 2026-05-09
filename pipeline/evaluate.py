"""
evaluate.py
Usage:
    python pipeline/evaluate.py \
        --model hstforu --scene data/drone/bike \
        --checkpoint checkpoints/hstforu_bike/best.pth
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from pipeline.dataset import DroneTestDataset


def compute_metrics(
    gt_labels: np.ndarray,
    scores: np.ndarray,
    threshold_percentile: float = 95,
) -> dict:
    """
    Args:
        gt_labels: binary ground truth (0=normal, 1=anomaly)
        scores: raw anomaly scores (higher = more anomalous)
        threshold_percentile: percentile of scores to use as detection threshold
    Returns:
        dict with auc, precision, recall, fpr, threshold
    """
    s_min, s_max = scores.min(), scores.max()
    norm = (scores - s_min) / (s_max - s_min + 1e-8)

    if len(np.unique(gt_labels)) < 2:
        auc = 0.5
    else:
        auc = roc_auc_score(gt_labels, norm)

    threshold = float(np.percentile(norm, threshold_percentile))
    binary_pred = (norm > threshold).astype(int)

    precision = precision_score(gt_labels, binary_pred, zero_division=0)
    recall = recall_score(gt_labels, binary_pred, zero_division=0)

    normal_mask = gt_labels == 0
    fpr = np.sum((binary_pred == 1) & normal_mask) / (
        np.sum(normal_mask) + 1e-8
    )

    return {
        "auc": round(float(auc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "fpr": round(float(fpr), 4),
        "threshold": round(float(threshold), 6),
    }


def smooth(scores: np.ndarray, window: int = 15) -> np.ndarray:
    """Rolling mean smoothing"""
    if window <= 1:
        return scores
    k = np.ones(window) / window
    return np.convolve(scores, k, mode="same")


def plot_score_timeline(
    scores, gt_labels, video_name, model_name, output_path
):
    """Save anomaly score timeline graph with GT overlay"""
    fig, ax = plt.subplots(figsize=(14, 4))
    frames = np.arange(len(scores))
    ax.plot(frames, scores, color="steelblue", lw=1, label="Anomaly Score")
    thresh = np.percentile(scores, 95)
    ax.axhline(thresh, color="orange", lw=1, ls="--", label="Threshold")
    in_anom, start = False, 0
    for i, lbl in enumerate(gt_labels):
        if lbl == 1 and not in_anom:
            start, in_anom = i, True
        elif lbl == 0 and in_anom:
            ax.axvspan(start, i, alpha=0.25, color="red", label="GT Anomaly")
            in_anom = False
    if in_anom:
        ax.axvspan(start, len(gt_labels), alpha=0.25, color="red")
    ax.set(
        xlabel="Frame",
        ylabel="Score",
        title=f"{model_name} | {video_name}",
    )
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _load_eval_model(model_type: str, checkpoint: str, device, clip_length: int):
    from pipeline.infer import load_model

    return load_model(
        model_type, checkpoint, device, clip_length=clip_length, img_size=256
    )


def scores_for_scene(
    model_type: str,
    scene_dir: str,
    checkpoint: str,
    device: torch.device,
    batch_size: int = 8,
    smooth_window: int = 15,
):
    """Compute per-(video, frame_index) scores on testing split."""
    clip_len = 5 if model_type == "hstforu" else 1
    ds = DroneTestDataset(
        scene_dir, "testing", clip_length=clip_len, img_size=256, stride=1
    )
    if len(ds) == 0:
        return np.array([]), np.array([]), {}

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    model = _load_eval_model(model_type, checkpoint, device, clip_len)
    model.eval()

    per_vid_scores = defaultdict(dict)
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                clips, vid_names, target_idxs, _ = batch
            else:
                clips, vid_names, target_idxs = batch
            clips = clips.to(device)
            if model_type == "conv_ae":
                x = clips.squeeze(1)
                recon = model(x)
                err = torch.mean((x - recon) ** 2, dim=(1, 2, 3))
            elif model_type == "astnet":
                x = clips.squeeze(1)
                recon, _ = model(x)
                err = torch.mean((x - recon) ** 2, dim=(1, 2, 3))
            else:
                inp = clips[:, :-1]
                tgt = clips[:, -1]
                pred = model(inp)
                err = torch.mean((pred - tgt) ** 2, dim=(1, 2, 3))
            for i in range(err.shape[0]):
                vn = vid_names[i]
                ti = int(target_idxs[i])
                per_vid_scores[vn][ti] = float(err[i].item())

    all_gt = []
    all_sc = []
    ann_index = {}
    for i in range(len(ds)):
        _, vn, ti, lbl = ds[i]
        if vn not in ann_index:
            ann_index[vn] = ds.annotations.get(vn)
        if vn in per_vid_scores and ti in per_vid_scores[vn]:
            all_gt.append(lbl)
            all_sc.append(per_vid_scores[vn][ti])

    gt = np.array(all_gt, dtype=np.int64)
    sc = np.array(all_sc, dtype=np.float64)
    if smooth_window > 1 and len(sc) > 0:
        sc = smooth(sc, smooth_window)
    return gt, sc, dict(per_vid_scores)


def run_cli(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gt, scores, _ = scores_for_scene(
        args.model,
        args.scene,
        args.checkpoint,
        device,
        smooth_window=args.smooth_window,
    )
    if len(gt) == 0:
        print("No test samples found. Check scene path and frames.")
        return
    metrics = compute_metrics(gt, scores, args.threshold_percentile)
    scene_name = os.path.basename(args.scene.rstrip("/\\"))
    print(json.dumps(metrics, indent=2))
    os.makedirs("outputs/benchmark", exist_ok=True)
    out_json = os.path.join(
        "outputs",
        "benchmark",
        f"{args.model}_{scene_name}_metrics.json",
    )
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["conv_ae", "astnet", "hstforu"])
    p.add_argument("--scene", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--threshold-percentile", type=float, default=95)
    p.add_argument("--smooth-window", type=int, default=15)
    run_cli(p.parse_args())
