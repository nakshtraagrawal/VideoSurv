"""
preprocess.py
Usage:
    python pipeline/preprocess.py --input data/raw/ --output data/drone/ --size 256
"""
import argparse
import os
from pathlib import Path

import cv2


def extract_frames(video_path: str, output_dir: str, size: int = 256):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (size, size))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(
            os.path.join(output_dir, f"{idx:04d}.jpg"),
            cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        )
        idx += 1
    cap.release()
    return idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Video file or directory of videos")
    p.add_argument("--output", required=True, help="Output frames root")
    p.add_argument("--size", type=int, default=256)
    args = p.parse_args()

    inp = args.input
    if os.path.isfile(inp):
        name = Path(inp).stem
        out = os.path.join(args.output, name)
        n = extract_frames(inp, out, args.size)
        print(f"Extracted {n} frames to {out}")
    else:
        for f in sorted(os.listdir(inp)):
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                fp = os.path.join(inp, f)
                name = Path(f).stem
                out = os.path.join(args.output, name)
                n = extract_frames(fp, out, args.size)
                print(f"{f}: {n} frames -> {out}")


if __name__ == "__main__":
    main()
