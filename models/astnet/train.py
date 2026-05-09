"""
Usage: python models/astnet/train.py --scene data/drone/bike --epochs 60
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from pipeline.dataset import DroneDataset
from models.astnet.model import ASTNet


def _maybe_wandb(name: str):
    try:
        import wandb

        wandb.init(project="autosurveil", name=name)
        return wandb
    except Exception:
        return None


def train(args):
    wb = _maybe_wandb(f"astnet_{os.path.basename(args.scene)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = DroneDataset(
        args.scene, "training", clip_length=1, img_size=256, stride=1
    )
    if len(ds) == 0:
        raise SystemExit(f"No training clips in {args.scene}")
    loader = DataLoader(
        ds,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = ASTNet(in_ch=1, base=args.base).to(device)
    opt = Adam(model.parameters(), lr=args.lr)
    sched = CosineAnnealingLR(opt, args.epochs)
    crit = nn.MSELoss()
    best = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for batch in loader:
            x = batch.squeeze(1).to(device)
            recon, _ = model(x)
            loss = crit(recon, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / max(len(loader), 1)
        sched.step()
        if wb:
            wb.log({"train_loss": avg, "epoch": epoch})
        print(f"[{epoch + 1}/{args.epochs}] loss={avg:.6f}")
        if avg < best:
            best = avg
            ckpt_dir = f"checkpoints/astnet_{os.path.basename(args.scene)}"
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(ckpt_dir, "best.pth"),
            )
    if wb:
        wb.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--base", type=int, default=48)
    p.add_argument("--num-workers", type=int, default=4)
    train(p.parse_args())
