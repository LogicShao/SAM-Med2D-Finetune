# B0 Stability Gate on AutoDL

## Scope

This note records the required three-epoch stability gate after the B0.5
runtime profile and before A0 evidence training.

## Run

- Run ID: `b06_stability_b16_w2_seed11171`
- Output root: `/root/autodl-tmp/runs/b06_stability_b16_w2_seed11171`
- Data view: `paper_dev_v1`
- Seed and dataset seed: `11171`
- Runtime: `batch_size=16`, `num_workers=2`, AMP off, cuDNN enabled,
  `cudnn_benchmark=false`
- Cache: `/root/autodl-tmp/cache/paper_dev_v1`, `cache_max_cases=8`
- Prompt/training baseline: positive slices only, zero negative prompt,
  no PJT, no hierarchy loss
- Snapshots written: `epoch_001.pth`, `epoch_002.pth`, `epoch_003.pth`

The SSH session reset after launch, but the training process completed and all
expected artifacts were written.

## Metrics

Metrics CSV:
`/root/autodl-tmp/runs/b06_stability_b16_w2_seed11171/logs/b06_stability_b16_w2_seed11171_adapter/metrics.csv`

| Epoch | Train steps | Samples/s | Peak alloc MiB | Train Dice | Slice val Dice | ET | TC | WT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 808 | 62.3629 | 6677.6416 | 0.7930 | 0.8254 | 0.7940 | 0.8746 | 0.8078 |
| 2 | 808 | 66.5240 | 6678.3916 | 0.8198 | 0.8356 | 0.8066 | 0.8811 | 0.8190 |
| 3 | 808 | 66.4588 | 6678.3916 | 0.8253 | 0.8355 | 0.8077 | 0.8808 | 0.8179 |

## Decision

The stability gate passes. The selected runtime can enter A0:

- `batch_size=16`
- `num_workers=2`
- AMP off
- cuDNN enabled
- `cudnn_benchmark=false`
- `cache_max_cases=8`
- `persistent_workers=false`
- `non_blocking_transfer=false`

Slice-level validation Dice is only a health signal. It must not select the
paper checkpoint. A0 still needs a five-epoch evidence trajectory with
immutable snapshots at epochs 1, 3, and 5 followed by raw patient-level 3D
checkpoint selection.
