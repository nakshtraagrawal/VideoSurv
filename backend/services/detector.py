"""
detector.py
Loads trained checkpoints and scores clips (same logic as pipeline/infer.py).
"""
from __future__ import annotations

import torch

from backend.services import state as app_state
from pipeline.infer import load_model


def _parse_checkpoint_key(folder_name: str) -> tuple[str, str] | None:
    for prefix in ("conv_ae", "hstforu", "astnet"):
        p = prefix + "_"
        if folder_name.startswith(p):
            return prefix, folder_name[len(p) :]
    return None


class Detector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: dict[str, torch.nn.Module] = {}

    def load(self, model_type: str, checkpoint_path: str, scene: str) -> None:
        clip_len = 5 if model_type == "hstforu" else 1
        m = load_model(
            model_type, checkpoint_path, self.device, clip_length=clip_len, img_size=256
        )
        key = f"{model_type}_{scene}"
        self.models[key] = m
        print(f"Loaded {key} from {checkpoint_path}")

    def load_from_checkpoint_path(self, checkpoint_path: str) -> None:
        import os

        folder = os.path.basename(os.path.dirname(checkpoint_path))
        parsed = _parse_checkpoint_key(folder)
        if not parsed:
            raise ValueError(f"Unrecognized checkpoint folder: {folder}")
        model_type, scene = parsed
        self.load(model_type, checkpoint_path, scene)

    def ensure_loaded(self, model_type: str, scene: str) -> str:
        key = f"{model_type}_{scene}"
        if key in self.models:
            return key
        import os

        ckpt = os.path.join("checkpoints", f"{model_type}_{scene}", "best.pth")
        if os.path.isfile(ckpt):
            self.load(model_type, ckpt, scene)
            return key
        raise FileNotFoundError(f"No checkpoint at {ckpt}")

    def score_clip(
        self, frames_tc_hw: torch.Tensor, model_type: str, scene: str
    ) -> float:
        """
        frames_tc_hw: (T, C, H, W) on CPU or device, normalized like training.
        Returns scalar anomaly score for the last frame in the window.
        """
        key = self.ensure_loaded(model_type, scene)
        model = self.models[key]
        model.eval()
        with torch.no_grad():
            if model_type == "hstforu":
                if frames_tc_hw.shape[0] < 5:
                    raise ValueError("HSTforU needs 5 frames")
                clip = frames_tc_hw[-5:].unsqueeze(0).to(self.device)
                pred = model(clip[:, :-1])
                tgt = clip[:, -1]
                err = torch.mean((pred - tgt) ** 2)
                return float(err.item())
            x = frames_tc_hw[-1:].unsqueeze(0).to(self.device)
            if model_type == "conv_ae":
                recon = model(x)
            else:
                recon, _ = model(x)
            err = torch.mean((x - recon) ** 2)
            return float(err.item())

    def score_video_folder(
        self,
        frames_dir: str,
        model_type: str,
        scene: str,
        threshold: float,
    ) -> tuple[list[float], str]:
        """Run sliding-window scoring; write mp4 via pipeline.infer.run."""
        import argparse
        import os

        from pipeline import infer

        ckpt = os.path.join("checkpoints", f"{model_type}_{scene}", "best.pth")
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(ckpt)
        out_mp4 = os.path.join(
            "outputs",
            "videos",
            f"infer_{scene}_{os.path.basename(frames_dir.rstrip('/'))}.mp4",
        )
        args = argparse.Namespace(
            frames=frames_dir,
            model=model_type,
            checkpoint=ckpt,
            output=out_mp4,
            threshold=threshold,
        )
        infer.run(args)
        scores_path = out_mp4.replace(".mp4", "_scores.npy")
        import numpy as np

        scores = np.load(scores_path).tolist() if os.path.isfile(scores_path) else []
        return scores, out_mp4


detector = Detector()
