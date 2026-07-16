# Y0 Paper V1 Dataset Verification - 2026-07-15

## Verdict

Status: `verified_pass`

The `brats_yolo_paper_v1_seed11171` dataset is fully verified against the parent
PRD protocol. All invariants hold: SHA256 match, correct case/slice counts,
train/val disjointness, test exclusion. The dataset is ready for Y2 retry.

## Verification Method

- Date: 2026-07-15
- Verification by: P0 execution audit (remote AutoDL read-only)
- Source: remote AutoDL `/root/autodl-tmp/datasets/brats_yolo_paper_v1_seed11171/`
- Python: `/root/autodl-tmp/envs/sam-med2d/bin/python`

## Source Split Integrity

| Split | Count | Overlap with train | Overlap with val | Overlap with test |
| --- | ---: | ---: | ---: | ---: |
| train | 875 | — | 0 | 0 |
| val | 187 | 0 | — | 0 |
| test | 189 | 0 | 0 | — |

- Seed: 11171 ✅
- Total unique IDs: 1251 ✅
- Ratios: 0.70/0.15/0.15 ✅

Source split manifest SHA256: `a1ba9d36accfe884ab8e0d043316643b2a61f2c144b059ac5d0e446d0ea9fe24`

## Dataset Manifest Verification

| Field | Expected | Actual | Match |
| --- | --- | --- | --- |
| SHA256 | `08eb0ac8f25697297d92b0bcb40510bdead5700ed63cf95357e649ecf4850b6f` | same | ✅ |
| schema_version | 1 | 1 | ✅ |
| dataset | brats_yolo_wt_box | brats_yolo_wt_box | ✅ |
| seed | 11171 | 11171 | ✅ |
| splits | ["train", "val"] | ["train", "val"] | ✅ |
| class_map | {0: Tumor} | {0: Tumor} | ✅ |
| modalities | t1ce, t2, flair | t1ce, t2, flair | ✅ |
| box_padding_ratio | 0.10 | 0.1 | ✅ |
| negative_to_positive_ratio | 0.333... | 0.3333333333333333 | ✅ |

## Export Integrity

| Split | Cases | Positive Slices | Negative Slices | Total Slices | Image Files | Label Files |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 875 | 57131 | 19049 | 76180 | 76180 | 76180 |
| val | 187 | 11950 | 17035 | 28985 | 28985 | 28985 |

- Train negative ratio: 19049 / 57131 = 0.3333... (1:3 constraint satisfied) ✅
- Val exports ALL slices (11950 pos + 17035 neg = 28985) — no negative sampling ✅
- Train/val case IDs are disjoint (verified via set intersection) ✅
- Test is NOT in `dataset_manifest.exports` ✅
- Test is NOT in `dataset_manifest.splits` ✅
- No test case ID appears in train or val case lists ✅

## Interrupted Y2 Run Evidence

The Y2 training run was interrupted and its evidence is preserved:

| Field | Value |
| --- | --- |
| Run ID | `y0_yolo11m_paper_v1_seed11171` |
| Run directory | `/root/autodl-tmp/runs/y0_yolo11m_paper_v1_seed11171` |
| Status | `interrupted` |
| Exit status | 143 (SIGTERM) |
| Started | 2026-07-14T13:49:49 UTC |
| Ended | 2026-07-14T13:58:22 UTC |
| Wall clock | ~513 seconds |
| Weights directory | **Empty** — no best.pt, no last.pt |
| Manifest | Present, records full provenance |

**Resume assessment**: Weights directory is empty. No checkpoint exists to resume
from. The documented retry strategy is `--resume true` which triggers
`retry_no_checkpoint` event in the manifest while preserving all historical
status entries.

## Conclusion

The Y0 dataset protocol is satisfied. Y2 retry is authorized with the
documented `retry_no_checkpoint` path. The interrupted run directory and
manifest must not be deleted.
