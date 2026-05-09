#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
SCENES=(bike highway crossroads)
for s in "${SCENES[@]}"; do
  if [[ -d "data/drone/$s/training/frames" ]]; then
    python models/conv_autoencoder/train.py --scene "data/drone/$s" --epochs 50
    python models/astnet/train.py --scene "data/drone/$s" --epochs 60
    python models/hstforu/train.py --scene "data/drone/$s" --epochs 100
  fi
done
