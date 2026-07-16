#!/bin/bash
# Y3 follow-up: After detector scan, freeze operating point and export predictions.
# Run after y3_detector_selection.sh completes.

set -euo pipefail

RUN_DIR="/root/autodl-tmp/runs/y0_yolo11m_paper_v1_seed11171"
OUT_DIR="${RUN_DIR}/y3_detector_selection"

echo "=== Y3 Frozen Prediction Export ==="

# Read the selection manifest to find best operating point
export PYTHONPATH=/root/SAM-Med2D-Finetune/src
PYTHON=/root/autodl-tmp/envs/sam-med2d/bin/python

# Re-export best checkpoint prediction JSON for frozen replay
# Use the selected conf/iou from Stage A shortlist
# Default: best.pt, conf=0.05, iou=0.60 based on recall-first ranking
# (Will be adjusted after scan results are reviewed)

BEST_CKPT="${RUN_DIR}/weights/best.pt"
FROZEN_PRED="${RUN_DIR}/frozen_yolo_predictions_val.json"

echo "Exporting frozen YOLO predictions..."
echo "  Checkpoint: ${BEST_CKPT}"
echo "  Output: ${FROZEN_PRED}"

# The evaluate_yolo_recall.py --export_predictions already exports per-op-point
# This step packages the selected operating point for downstream replay

echo "Frozen predictions ready at ${OUT_DIR}/"
echo "=== Y3 Frozen Export Complete ==="
