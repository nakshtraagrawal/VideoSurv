#!/usr/bin/env bash
# Drone-Anomaly: https://github.com/Jin-Pu/Drone-Anomaly
# Download the dataset from the OneDrive link in that README, then extract so you have:
#   data/drone/{scene}/training/frames/{video}/*.jpg
#   data/drone/{scene}/testing/frames/{video}/*.jpg
#   data/drone/{scene}/annotation/{video}.npy

set -euo pipefail
echo "1. Open https://github.com/Jin-Pu/Drone-Anomaly"
echo "2. Use the OneDrive link in README to download the release archive."
echo "3. Extract into ./data/drone/ preserving the scene folders (bike, highway, ...)."
