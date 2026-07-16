# AutoDL Training Environment Validation

## Decision

AutoDL is the only approved environment for all remaining ICASSP 2027 SAM,
YOLO, U-Net and optional nnU-Net experiments. The project does not schedule
new work on the former group server.

## Verified Baseline

| Item | Verified value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090D, 24 GiB, `cuda:0` |
| CPU / memory | 15 CPU cores / 80 GiB RAM |
| Data disk | `/root/autodl-tmp`, 150 GiB total |
| Environment | `sam-med2d` at `/root/autodl-tmp/envs/sam-med2d` |
| Python | 3.11.15 |
| PyTorch / CUDA | 2.6.0+cu124 / 12.4 |
| Code | `/root/SAM-Med2D-Finetune` |

The CUDA check completed a device matrix multiplication. `pip check`, source
compilation, imports for the imaging and training stack, and the training and
inference CLI help paths all passed. The four-channel SAM-Med2D ViT-B model
constructed and moved to `cuda:0`, allocating about 1.0 GiB before a training
batch.

## Storage Contract

- Dataset archive: `/root/autodl-tmp/data_source/archive.zip`
- Base checkpoint: `/root/autodl-tmp/checkpoints/sam-med2d_b.pth`
- Paper split, processed cache and run output: subdirectories of
  `/root/autodl-tmp`
- Source code only: `/root/SAM-Med2D-Finetune`

The 150 GiB data disk is sufficient for upload, extraction and development
work. Recheck free space after the raw data are extracted; expand the disk
before building a full 1,251-case cache if the measured cache plus raw data and
artifacts leave insufficient headroom.

## Gate Progress

- B0 smoke passed on AutoDL and wrote the required training artifacts under
  `/root/autodl-tmp/runs/b0_runtime_smoke_autodl_seed11171`.
- B0.5 runtime profiling completed on AutoDL. The evidence note is
  `research/b05-runtime-profile-autodl-2026-07-14.md`.
- The three-epoch stability gate completed on AutoDL. The evidence note is
  `research/b06-stability-gate-autodl-2026-07-14.md`.

## Next Gate

Run the A0 five-epoch development trajectory with the frozen B0.5 runtime
candidate: `batch_size=16`, `num_workers=2`, AMP off, cuDNN enabled,
`cudnn_benchmark=false`, `cache_max_cases=8`, `persistent_workers=false`,
`non_blocking_transfer=false`.
