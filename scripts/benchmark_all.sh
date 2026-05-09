#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
for ckpt in checkpoints/*/best.pth; do
  [[ -f "$ckpt" ]] || continue
  dir=$(basename "$(dirname "$ckpt")")
  case "$dir" in
    conv_ae_*)
      mt=conv_ae
      scene="${dir#conv_ae_}"
      ;;
    astnet_*)
      mt=astnet
      scene="${dir#astnet_}"
      ;;
    hstforu_*)
      mt=hstforu
      scene="${dir#hstforu_}"
      ;;
    *) continue ;;
  esac
  python pipeline/evaluate.py --model "$mt" --scene "data/drone/$scene" --checkpoint "$ckpt"
done
