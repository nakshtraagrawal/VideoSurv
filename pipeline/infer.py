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
import argparse
import os
from collections import deque

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T


def load_model(model_type: str, checkpoint: str, device, clip_length=5, img_size=256):
    if model_type == "conv_ae":
        from models.conv_autoencoder.model import ConvAutoencoder

        m = ConvAutoencoder()
    elif model_type == "astnet":
        from models.astnet.model import ASTNet

        m = ASTNet()
    elif model_type == "hstforu":
        from models.hstforu.model import HSTforU

        m = HSTforU(
            encoder_name="pvt_v2_b2",
            clip_length=clip_length,
            img_size=img_size,
            in_channels=1,
        )
    else:
        raise ValueError(model_type)
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    m.load_state_dict(state, strict=True)
    return m.to(device).eval()


def make_heatmap_overlay(frame_bgr, score_map_8x8, alpha=0.4):
    h, w = frame_bgr.shape[:2]
    sm = cv2.resize(
        score_map_8x8.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
    )
    sm = cv2.GaussianBlur(sm, (21, 21), 0)
    if sm.max() > sm.min():
        sm = ((sm - sm.min()) / (sm.max() - sm.min()) * 255).astype(np.uint8)
    else:
        sm = np.zeros((h, w), dtype=np.uint8)
    heatmap = cv2.applyColorMap(sm, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame_bgr, 1 - alpha, heatmap, alpha, 0)


def add_ui_overlay(frame, score, threshold, frame_idx, is_anomaly):
    h, w = frame.shape[:2]
    bar_y = h - 28
    cv2.rectangle(frame, (10, bar_y), (w - 10, bar_y + 18), (40, 40, 40), -1)
    denom = max(threshold * 2, 1e-8)
    fill = int(min(score, threshold * 2) / denom * (w - 20))
    color = (0, 60, 255) if is_anomaly else (0, 200, 80)
    cv2.rectangle(frame, (10, bar_y), (10 + fill, bar_y + 18), color, -1)
    cv2.putText(
        frame,
        f"SCORE: {score:.5f}  THR: {threshold:.5f}",
        (14, bar_y + 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 255, 255),
        1,
    )
    if is_anomaly:
        cv2.rectangle(frame, (0, 0), (w, 36), (0, 0, 180), -1)
        cv2.putText(
            frame,
            "ANOMALY DETECTED",
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
    cv2.putText(
        frame,
        f"#{frame_idx:05d}",
        (w - 80, h - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (180, 180, 180),
        1,
    )
    return frame


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_len = 5 if args.model == "hstforu" else 1
    model = load_model(
        args.model, args.checkpoint, device, clip_length=clip_len, img_size=256
    )

    transform = T.Compose(
        [
            T.Resize((256, 256)),
            T.Grayscale(1),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ]
    )

    frames_files = sorted(
        [f for f in os.listdir(args.frames) if f.endswith(".jpg")]
    )
    if not frames_files:
        raise SystemExit(f"No .jpg frames in {args.frames}")

    sample_frame = cv2.imread(os.path.join(args.frames, frames_files[0]))
    H, W = sample_frame.shape[:2]

    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H)
    )

    clip_buf = deque(maxlen=5)
    score_buf = deque(maxlen=15)
    all_scores = []

    for i, fname in enumerate(frames_files):
        bgr = cv2.imread(os.path.join(args.frames, fname))
        if bgr is None:
            continue
        t = transform(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        clip_buf.append(t)

        score = 0.0
        score_map = np.zeros((8, 8))

        if args.model == "hstforu" and len(clip_buf) == 5:
            with torch.no_grad():
                clip = torch.stack(list(clip_buf)).unsqueeze(0).to(device)
                pred = model(clip[:, :-1])
                target = clip[:, -1]
                err_map = (
                    torch.mean((pred - target) ** 2, dim=1).squeeze().cpu().numpy()
                )
                score = float(err_map.mean())
                score_map = cv2.resize(err_map, (8, 8))
        elif args.model != "hstforu" and len(clip_buf) >= 1:
            with torch.no_grad():
                x = t.unsqueeze(0).to(device)
                if args.model == "conv_ae":
                    recon = model(x)
                else:
                    recon, _ = model(x)
                score = float(torch.mean((x - recon) ** 2).item())
                err = (x - recon).abs().squeeze().cpu().numpy()
                score_map = cv2.resize(err, (8, 8))

        score_buf.append(score)
        smoothed = float(np.mean(score_buf)) if score_buf else 0.0
        all_scores.append(smoothed)

        out = make_heatmap_overlay(bgr.copy(), score_map)
        out = add_ui_overlay(
            out,
            smoothed,
            args.threshold,
            i,
            smoothed > args.threshold,
        )
        writer.write(out)

        if i % 200 == 0:
            print(f"  {i}/{len(frames_files)}  score={smoothed:.5f}")

    writer.release()
    scores_path = args.output.replace(".mp4", "_scores.npy")
    np.save(scores_path, np.array(all_scores))
    print(f"Done. Video: {args.output}  Scores: {scores_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True)
    p.add_argument(
        "--model", required=True, choices=["conv_ae", "astnet", "hstforu"]
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="outputs/videos/result.mp4")
    p.add_argument("--threshold", type=float, default=0.015)
    run(p.parse_args())
