# P0 Execution Plan - 2026-07-15

## Current State

### Completed
- [x] PRD protocol fixes: seed 42→11171 unified, A3 promotion rule synced to Fallback A-D
- [x] Y0 verification: dataset manifest SHA256 confirmed, 875/187 cases, test excluded
- [x] PJT implementation: `multitask_dataset.py` + `train_multitask.py` CLI flags
- [x] Hierarchy loss implementation: training + validation loops
- [x] 2D U-Net model: `models/unet2d.py` (7.76M params) + `training/train_unet2d.py`
- [x] Code deployed to AutoDL `/root/SAM-Med2D-Finetune/`

### Running
- **Y2**: YOLO11m training (retry), PID 2419, ~6min elapsed, epoch 1/100
  - Speed: ~2.7 it/s, ~29 min/epoch train + ~10 min val
  - ETA: 20-65 hours depending on convergence
  - Run dir: `/root/autodl-tmp/runs/y0_yolo11m_paper_v1_seed11171`
  - GPU: 4.2GB / 24GB

### Blocked (waiting for Y2 → Y3)
- Y3: Detector selection scan
- A0: Raw 3D dev-val evaluation with YOLO prompts
- A1/A2/A3: Development ablation training
- A4: Conditional Tree-Min (if hierarchy claim triggered)
- 2D U-Net: Training

## Post-Y2 Execution Sequence

### Step 1: Verify Y2 Completion
```bash
# Check manifest status, weights, checkpoints
cat /root/autodl-tmp/runs/y0_yolo11m_paper_v1_seed11171/manifest.json | python -c "import json,sys; m=json.load(sys.stdin); print(m['status'], m['exit_status'])"
ls -la /root/autodl-tmp/runs/y0_yolo11m_paper_v1_seed11171/weights/
sha256sum /root/autodl-tmp/runs/y0_yolo11m_paper_v1_seed11171/weights/best.pt
sha256sum /root/autodl-tmp/runs/y0_yolo11m_paper_v1_seed11171/weights/last.pt
```

### Step 2: Y3 Detector Selection
```bash
PYTHONPATH=/root/SAM-Med2D-Finetune/src /root/autodl-tmp/envs/sam-med2d/bin/python \
  -m sam_med2d_finetune.tools.evaluate_yolo_recall \
  --model /root/autodl-tmp/runs/y0_yolo11m_paper_v1_seed11171/weights/best.pt \
  --data /root/autodl-tmp/datasets/brats_yolo_paper_v1_seed11171/data.yaml \
  --split val --conf_values 0.001,0.003,0.005,0.01,0.03,0.05,0.10 \
  --iou 0.60 --imgsz 320 --device 0 --max_det 1 --batch 64 \
  --ultralytics_dir /root/autodl-tmp/.ultralytics \
  --out_dir /root/autodl-tmp/runs/y0_yolo11m_paper_v1_seed11171/y3_detector_selection
```
Then repeat for last.pt.

Review scan results, apply Stage A ranking (lexicographic: missed cases → missed slices → max consecutive misses → coverage recall → FP rate → area ratio), select up to 2 candidates, run Stage B with A0 on 43-case dev-val.

### Step 3: Freeze Detector Operating Point
Export frozen prediction JSON for selected (checkpoint, conf, iou). Bind to checkpoint + dataset manifest SHA256.

### Step 4: A0 Raw 3D Evaluation (43-case dev-val)
```bash
PYTHONPATH=/root/SAM-Med2D-Finetune/src /root/autodl-tmp/envs/sam-med2d/bin/python \
  -m sam_med2d_finetune.inference.batch_validate \
  --cases_root /root/autodl-tmp/data_brats_paper_dev_v1/val \
  --output_root /root/autodl-tmp/runs/a0_adapter_dev_seed11171/eval_yolo_raw \
  --sam_checkpoint /root/autodl-tmp/checkpoints/sam-med2d_b.pth \
  --finetuned_checkpoint /root/autodl-tmp/runs/a0_adapter_dev_seed11171/models/a0_adapter_dev_seed11171_adapter/best_model.pth \
  --finetune_method adapter \
  --prompt_mode frozen_yolo_box \
  --yolo_predictions <frozen_pred_path> \
  --postprocess false --threshold 0.5 --use_amp true \
  --device cuda:0 --image_size 256
```

Also run with `--prompt_mode upper_bound` for oracle baseline and `--prompt_mode full_image_box` for no-localization reference.

### Step 5: A1/A2/A3 Development Training (200/43, seed 11171)

All use same dev cache, same epoch budget, same checkpoint proxy.

**A1** (PJT only):
```bash
PYTHONPATH=src /root/autodl-tmp/envs/sam-med2d/bin/python \
  -m sam_med2d_finetune.training.train_multitask \
  --finetune_method adapter --train_data_path /root/autodl-tmp/data_brats_paper_dev_v1/train \
  --val_data_path /root/autodl-tmp/data_brats_paper_dev_v1/val \
  --work_dir /root/autodl-tmp/runs --run_name a1_pjt_dev \
  --epochs 5 --batch_size 16 --image_size 256 \
  --sam_checkpoint /root/autodl-tmp/checkpoints/sam-med2d_b.pth \
  --cache_root /root/autodl-tmp/cache/paper_dev_v1 --cache_max_cases 8 \
  --negative_to_positive_ratio 0.3333333333333333 --negative_prompt_box random \
  --pjt true --pjt_translate_max 0.10 --pjt_scale_min 0.85 --pjt_scale_max 1.15 \
  --pjt_miss_prob 0.0 --hierarchy_loss false \
  --seed 11171 --deterministic true --device cuda:0 --use_amp false \
  --num_workers 2 --save_epochs 1 3 5
```
(Note: PJT ranges must be derived from YOLO error distributions per PRD. Above are initial defaults.)

**A2** (hierarchy only):
```bash
# Same as A0 base but with --hierarchy_loss true --lambda_hier <selected_value>
# negative_to_positive_ratio 0.0 --negative_prompt_box zero --pjt false
```

**A3** (PJT + hierarchy):
```bash
# Combine A1 + A2 settings
```

### Step 6: 2D U-Net Training (dev screen, seed 11519)
```bash
PYTHONPATH=src /root/autodl-tmp/envs/sam-med2d/bin/python \
  -m sam_med2d_finetune.training.train_unet2d \
  --train_data_path /root/autodl-tmp/data_brats_paper_dev_v1/train \
  --val_data_path /root/autodl-tmp/data_brats_paper_dev_v1/val \
  --work_dir /root/autodl-tmp/runs --run_name unet2d_dev_seed11519 \
  --epochs 100 --batch_size 16 --image_size 256 \
  --cache_root /root/autodl-tmp/cache/paper_dev_v1 --cache_max_cases 8 \
  --negative_to_positive_ratio 0.3333333333333333 \
  --seed 11519 --dataset_seed 11519 --deterministic true \
  --device cuda:0 --use_amp false --num_workers 2
```

## Disk Space

- `/root/autodl-tmp`: 150G total, 106G used, 45G free (71%)
- Full paper_v1 cache not yet built — will need for full-data runs
- Monitor disk before building full cache

## Risk Register

1. **Y2 does not converge**: If early stopping at epoch 20+ shows poor results, one documented recall-oriented retraining is allowed.
2. **Y3 fails recall gate**: If no candidate passes (missed cases, coverage<0.98), permit one documented retraining.
3. **A0 dev-val Dice too low**: Diagnose with oracle prompts first; if gap is small, proceed with analysis route.
4. **Disk space**: If <10GB free after Y2+Y3+A0, build full cache later or expand disk.
5. **All method fallbacks fail**: Switch to analysis-only paper track per parent PRD.
