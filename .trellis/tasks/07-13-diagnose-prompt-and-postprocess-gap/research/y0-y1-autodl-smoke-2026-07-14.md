# Y0-Y1 AutoDL Smoke Record - 2026-07-14

## Scope

This record covers the first YOLO prompt-path gate after moving training to
AutoDL. The goal was not detector quality; it was to verify that the frozen
BraTS split can be materialized into YOLO format and that YOLO11m can run one
real train/val epoch on the AutoDL 4090D environment without relying on remote
model download.

## Local Checkpoint Transfer

- Local download used HTTP proxy `127.0.0.1:7897`.
- Local file: `downloads/yolo11m.pt`.
- Remote file: `/root/autodl-tmp/checkpoints/yolo11m.pt`.
- Transfer method: project `autodl-rsync-sync` skill, `upload-file`.
- SHA256: `d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95`.
- Size: `40684120` bytes.

## Y0 Dataset Smoke

- Source split root: `/root/autodl-tmp/data_brats_paper_dev_v1`.
- Output: `/root/autodl-tmp/datasets/brats_yolo_dev_smoke_seed11171_20260714`.
- Seed: `11171`.
- Case cap: `--max_cases_per_split 2`.
- Train export:
  - cases: `2`
  - source positive slices: `122`
  - source negative slices: `188`
  - exported positive slices: `122`
  - exported negative slices: `40`
  - exported slices: `162`
- Val export:
  - cases: `2`
  - source positive slices: `125`
  - source negative slices: `185`
  - exported positive slices: `125`
  - exported negative slices: `185`
  - exported slices: `310`
- Disk footprint: `11M`.
- Source split manifest SHA256:
  `775f1de6f8021aab60ad8d884ad4de8936e61c3cca5bbb5fdfb28a2573129982`.

Integrity checks passed:

- train images and labels: `162 / 162`
- val images and labels: `310 / 310`
- train non-empty labels: `122`, empty labels: `40`
- val non-empty labels: `125`, empty labels: `185`
- `data.yaml` class map: `{0: Tumor}`
- manifest seed: `11171`

## Y1 Runtime Smoke

Command shape:

```bash
PYTHONPATH=src /root/autodl-tmp/envs/sam-med2d/bin/python \
  -m sam_med2d_finetune.tools.train_yolo \
  --data /root/autodl-tmp/datasets/brats_yolo_dev_smoke_seed11171_20260714/data.yaml \
  --model /root/autodl-tmp/checkpoints/yolo11m.pt \
  --epochs 1 --imgsz 320 --batch 16 --device 0 --workers 4 \
  --patience 20 --project /root/autodl-tmp/runs \
  --name y1_yolo11m_dev_smoke_seed11171_20260714 \
  --seed 11171 --amp false --cache false --plots false --save_period -1 \
  --ultralytics_dir /root/autodl-tmp/ultralytics
```

Environment:

- Ultralytics: `8.4.95`
- Python: `3.11.15`
- PyTorch: `2.6.0+cu124`
- GPU: `NVIDIA GeForce RTX 4090 D`, 24081 MiB

Observed smoke result:

- Status: completed.
- Peak YOLO training GPU memory: approximately `4.29G`.
- Train iterations: `11`.
- Validation images: `310`.
- Validation instances: `125`.
- Epoch time in `results.csv`: `3.58316` seconds.
- Metrics after one epoch:
  - precision: `0.08336`
  - recall: `0.112`
  - mAP50: `0.04129`
  - mAP50-95: `0.01042`
- Run footprint: `78M`.
- Checkpoints:
  - best: `/root/autodl-tmp/runs/y1_yolo11m_dev_smoke_seed11171_20260714/weights/best.pt`
  - last: `/root/autodl-tmp/runs/y1_yolo11m_dev_smoke_seed11171_20260714/weights/last.pt`
- Checkpoint hashes:
  - best: `3241866b43167ffea20c8fa6a16a04cdb5b0da04934d8eae429bdc06897e7a22`
  - last: `1ea500c46ef43370aa3dbd6cbdd74ab656539cc1deecfe74eb2b9cbfbf600a9c`

## Space Check

Before continuing to Y2, `/root/autodl-tmp` remained at:

- total: `150G`
- used: `103G`
- available: `48G`
- usage: `69%`

No expansion is needed for the next dataset-materialization step. If space
pressure appears later, first inspect and consider cleanup candidates under
old B0/B0.5/B0.6 profiling runs, `pip_cache`, and extraction staging data
before requesting disk expansion.

## Next Step

Proceed to Y0 full materialization for the frozen `paper_v1` split, then run
the single Y2 YOLO11m training job using the local checkpoint path:

`/root/autodl-tmp/checkpoints/yolo11m.pt`

Keep seed `11171`, `imgsz=320`, `batch=16`, `workers=4`, and `amp=false`
unless a reproducible runtime failure requires a documented fallback.
