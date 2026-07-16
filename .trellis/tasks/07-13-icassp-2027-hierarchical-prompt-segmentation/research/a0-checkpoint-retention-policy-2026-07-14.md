# A0 Checkpoint Retention Policy

Date: 2026-07-14

## Decision

The active AutoDL training path should keep only the retained
`best_model.pth` for SAM Adapter development runs. Standard A0/A1/A2/A3
launches must not pass `--save_epochs`.

This is a storage-control decision. It changes the development workflow from
the earlier epoch 1/3/5 snapshot plan to a single retained checkpoint plus
complete metrics/provenance records.

## Rationale

- Adapter checkpoints are about 1.09 GB each.
- Keeping epoch snapshots across A0/A1/A2/A3 and later seeds grows the data
  directory quickly without adding enough value at the current stage.
- `metrics.csv`, resolved launch arguments, logs, checkpoint hash, and raw 3D
  evaluation output provide the required provenance for development decisions.

## Caveat

The current training entry point selects `best_model.pth` using slice-level
validation Dice. This is acceptable as a development proxy under the
storage-limited policy, but it is not yet the formal paper checkpoint-selection
endpoint. Paper-facing claims must either:

1. explicitly report this selection rule, or
2. replace it with raw patient-level 3D development selection before locked
   test evaluation.

## A0 Run State

Remote run:

```text
/root/autodl-tmp/runs/a0_adapter_dev_seed11171
```

Completed metrics:

| epoch | val Dice | ET | TC | WT |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.8253 | 0.7939 | 0.8744 | 0.8075 |
| 2 | 0.8355 | 0.8065 | 0.8810 | 0.8190 |
| 3 | 0.8353 | 0.8076 | 0.8804 | 0.8179 |
| 4 | 0.8378 | 0.8119 | 0.8812 | 0.8203 |
| 5 | 0.8379 | 0.8144 | 0.8761 | 0.8232 |

Peak memory was about 6678 MiB allocated and 7104 MiB reserved.

Retained checkpoint:

```text
best_model.pth
sha256: 97dbb0bca186449c280d15f61e304b6fb797c8e6fff722cb6d31ea4f0b0a7591
```

Remote artifacts before cleanup included:

```text
best_model.pth
epoch_001.pth
epoch_003.pth
epoch_005.pth
```

The three `epoch_*.pth` files are legacy diagnostic snapshots from the previous
protocol. They should not be used by later A1/A2/A3 launches. Removing them
requires explicit user confirmation because it is a file deletion operation.
They occupy about 3.1 GB in total.

Cleanup completed after explicit user confirmation:

```text
deleted: epoch_001.pth
deleted: epoch_003.pth
deleted: epoch_005.pth
retained: best_model.pth
```

Post-cleanup model directory size is about 1.1 GB, and `/root/autodl-tmp`
reported about 48 GB available.

## Next Launch Contract

Use this default for A1/A2/A3 unless a new PRD revision says otherwise:

```text
--epochs 5
--batch_size 16
--num_workers 2
--use_amp false
--disable_cudnn false
--cudnn_benchmark false
--seed 11171
--dataset_seed 11171
```

Do not include:

```text
--save_epochs 1 3 5
```
