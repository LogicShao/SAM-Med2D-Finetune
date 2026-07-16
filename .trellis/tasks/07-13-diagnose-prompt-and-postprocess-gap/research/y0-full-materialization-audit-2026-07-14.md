# Y0 Full YOLO Dataset Materialization Audit - 2026-07-14

## Verdict

Status: `succeeded`

The complete frozen `paper_v1` train/val YOLO dataset was materialized on
AutoDL without exporting test cases. The resulting dataset satisfies the Y0
Dataset Protocol and is ready as the data input for Y2 YOLO11m training.

## Scope

- Task: `.trellis/tasks/07-13-diagnose-prompt-and-postprocess-gap`
- Protocol stage: `Y0 Dataset Protocol`
- Source split root: `/root/autodl-tmp/data_brats_paper_v1`
- Output dataset: `/root/autodl-tmp/datasets/brats_yolo_paper_v1_seed11171`
- AutoDL code root: `/root/SAM-Med2D-Finetune`
- Environment command used: `/root/autodl-tmp/envs/sam-med2d/bin/python`
- No Y2 training was started.
- No model or evaluation logic was modified.
- No git commit was executed.

## Command

```bash
PYTHONPATH=src /root/autodl-tmp/envs/sam-med2d/bin/python \
  -m sam_med2d_finetune.tools.prepare_yolo_data \
  --split_root /root/autodl-tmp/data_brats_paper_v1 \
  --out_dir /root/autodl-tmp/datasets/brats_yolo_paper_v1_seed11171 \
  --splits train val \
  --seed 11171 \
  --negative_to_positive_ratio 0.3333333333333333 \
  --box_padding_ratio 0.10
```

No `--clean` and no `--max_cases_per_split` were used.

## Source Split Audit

The source split was verified before materialization:

| Split | Manifest Count | Disk Count | Manifest Matches Disk |
| --- | ---: | ---: | --- |
| train | 875 | 875 | true |
| val | 187 | 187 | true |
| test | 189 | 189 | true |

Overlap checks:

| Check | Count |
| --- | ---: |
| train ∩ val | 0 |
| train ∩ test | 0 |
| val ∩ test | 0 |

Source split manifest SHA256:

`a1ba9d36accfe884ab8e0d043316643b2a61f2c144b059ac5d0e446d0ea9fe24`

## Output Dataset Audit

Output files:

- `dataset_manifest.json`: present
- `data.yaml`: present
- Remote audit artifact: `/root/autodl-tmp/datasets/brats_yolo_paper_v1_seed11171/y0_full_audit.json`

Dataset manifest SHA256:

`08eb0ac8f25697297d92b0bcb40510bdead5700ed63cf95357e649ecf4850b6f`

Data YAML SHA256:

`bb420ee46da3fe062bac4d5dec59630eb72be115d2312b830d4d9fa28ad4a756`

Frozen parameters:

| Field | Value |
| --- | --- |
| splits | `train`, `val` |
| seed | `11171` |
| negative_to_positive_ratio | `0.3333333333333333` |
| box_padding_ratio | `0.1` |
| class map | `{0: Tumor}` |
| modalities | `t1ce`, `t2`, `flair` |
| normalization | `per_volume_nonzero_minmax_v1` |

Export counts:

| Split | Cases | Images | Labels | Non-empty Labels | Empty Labels | Image/Label Diff | Invalid Lines | Invalid Values |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 875 | 76180 | 76180 | 57131 | 19049 | 0 | 0 | 0 |
| val | 187 | 28985 | 28985 | 11950 | 17035 | 0 | 0 | 0 |

Manifest slice counts:

| Split | Source Positive | Source Negative | Exported Positive | Exported Negative | Exported Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 57131 | 78494 | 57131 | 19049 | 76180 |
| val | 11950 | 17035 | 11950 | 17035 | 28985 |

Interpretation:

- Train exported every positive slice and a deterministic per-case negative
  subset capped by the frozen 1:3 ratio.
- Val exported every slice from all 187 validation cases.
- Empty label counts match exported background slices.
- All non-empty YOLO label rows use class `0` and normalized coordinate values
  within `[0, 1]`.
- Exported train/val cases exactly match the corresponding source manifest
  partitions.
- Exported train/val cases contain no test cases.

## Runtime And Space

- Output directory size: `2.8G`
- `/root/autodl-tmp` after audit: `150G` total, `106G` used, `45G` available,
  `71%` used.
- Exact wrapper stdout was lost because the foreground SSH session was
  interrupted; subsequent polling confirmed the original remote process kept
  running and completed. Observed process `ETIME` polling indicates an
  approximate materialization time of 25 minutes.

No disk expansion is required before Y2. If space pressure appears later,
first inspect old profiling runs, pip cache, and extraction staging data before
requesting expansion.

## Y2 Readiness

Y2 has the required dataset condition:

`/root/autodl-tmp/datasets/brats_yolo_paper_v1_seed11171/data.yaml`

Y2 should use the already transferred local checkpoint:

`/root/autodl-tmp/checkpoints/yolo11m.pt`

Training was intentionally not started in this step.
