#!/bin/bash
# Y3: YOLO detector selection scan
# Evaluates best.pt and last.pt on 187 val cases, scans confidence thresholds,
# exports prediction JSONs, selects operating point.
# Run AFTER Y2 completes and saves best.pt/last.pt.

set -euo pipefail

RUN_DIR="/root/autodl-tmp/runs/y0_yolo11m_paper_v1_seed11171"
DATA="/root/autodl-tmp/datasets/brats_yolo_paper_v1_seed11171/data.yaml"
OUT_DIR="${RUN_DIR}/y3_detector_selection"

echo "=== Y3 Detector Selection ==="
echo "Run dir: ${RUN_DIR}"
echo "Started at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Verify checkpoints exist
for CKPT in best.pt last.pt; do
    CKPT_PATH="${RUN_DIR}/weights/${CKPT}"
    if [ ! -f "${CKPT_PATH}" ]; then
        echo "FATAL: ${CKPT_PATH} not found. Y2 training may not have completed successfully."
        exit 1
    fi
    echo "Checkpoint ${CKPT}: $(sha256sum ${CKPT_PATH} | cut -d' ' -f1)"
done

export PYTHONPATH=/root/SAM-Med2D-Finetune/src
PYTHON=/root/autodl-tmp/envs/sam-med2d/bin/python

# Scan both checkpoints with all confidence thresholds
CONF_VALUES="0.001,0.003,0.005,0.01,0.03,0.05,0.10"

for CKPT in best.pt last.pt; do
    CKPT_PATH="${RUN_DIR}/weights/${CKPT}"
    echo ""
    echo "--- Scanning ${CKPT} ---"

    ${PYTHON} -m sam_med2d_finetune.tools.evaluate_yolo_recall \
        --model "${CKPT_PATH}" \
        --data "${DATA}" \
        --split val \
        --conf_values "${CONF_VALUES}" \
        --iou 0.60 \
        --imgsz 320 \
        --device 0 \
        --max_det 1 \
        --batch 64 \
        --ultralytics_dir /root/autodl-tmp/.ultralytics \
        --out_dir "${OUT_DIR}/${CKPT%.pt}"
done

echo ""
echo "=== Y3 Complete at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
