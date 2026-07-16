# Experiment Report: `y1_yolo11m_dev_smoke_seed11171_20260714`

## Verdict

| Field | Value |
| --- | --- |
| Status | `succeeded` |
| Decision | `promote` |
| Profile | `smoke` |
| Owning task | `.trellis/tasks/07-13-diagnose-prompt-and-postprocess-gap` |
| Root research task | `.trellis/tasks/07-13-icassp-2027-hierarchical-prompt-segmentation` |

**Finding:** The AutoDL YOLO11m runtime gate passed with local-checkpoint loading, real BraTS-derived YOLO data, one backward training epoch, and validation on complete smoke volumes; this supports moving to full Y0 materialization and Y2 training, but makes no detector-quality claim.

## Reproducibility

| Field | Value | Evidence |
| --- | --- | --- |
| Purpose | Verify Y0 data materialization and Y1 YOLO11m train/val runtime health on AutoDL. | `.trellis/tasks/07-13-diagnose-prompt-and-postprocess-gap/prd.md`; `.trellis/tasks/07-13-diagnose-prompt-and-postprocess-gap/research/y0-y1-autodl-smoke-2026-07-14.md` |
| Code revision / dirty state | `11756be865507bc16c43c301106b3faf7f9a92cf`, dirty worktree with 36 porcelain entries. | `git rev-parse HEAD`; `git status --porcelain` |
| Data view / cases | `data_brats_paper_dev_v1` smoke cap, train 2 cases and val 2 cases; train 162 exported slices, val 310 exported slices. | `/root/autodl-tmp/datasets/brats_yolo_dev_smoke_seed11171_20260714/dataset_manifest.json`; `.trellis/tasks/07-13-diagnose-prompt-and-postprocess-gap/research/y0-y1-autodl-smoke-2026-07-14.md` |
| Split manifest / SHA-256 | `775f1de6f8021aab60ad8d884ad4de8936e61c3cca5bbb5fdfb28a2573129982`. | `/root/autodl-tmp/datasets/brats_yolo_dev_smoke_seed11171_20260714/dataset_manifest.json` |
| Method / prompt / postprocess | One-class WT YOLO detector smoke using `seg > 0` boxes; no SAM prompt replay or postprocess evaluation. | `/root/autodl-tmp/datasets/brats_yolo_dev_smoke_seed11171_20260714/dataset_manifest.json`; `/root/autodl-tmp/runs/y1_yolo11m_dev_smoke_seed11171_20260714/train_config.json` |
| Seeds | `11171`. | `/root/autodl-tmp/runs/y1_yolo11m_dev_smoke_seed11171_20260714/train_config.json` |
| Command / resolved config | `imgsz=320`, `batch=16`, `workers=4`, `device=0`, `amp=false`, `epochs=1`, `save_period=-1`. | `/root/autodl-tmp/runs/y1_yolo11m_dev_smoke_seed11171_20260714/train_config.json` |
| Checkpoints / hashes | Base `yolo11m.pt`: `d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95`; smoke best: `3241866b43167ffea20c8fa6a16a04cdb5b0da04934d8eae429bdc06897e7a22`; smoke last: `1ea500c46ef43370aa3dbd6cbdd74ab656539cc1deecfe74eb2b9cbfbf600a9c`. | `sha256sum /root/autodl-tmp/checkpoints/yolo11m.pt /root/autodl-tmp/runs/y1_yolo11m_dev_smoke_seed11171_20260714/weights/*.pt` |
| Runtime / environment | Ultralytics `8.4.95`, Python `3.11.15`, PyTorch `2.6.0+cu124`, RTX 4090D; peak YOLO training memory about `4.29G`; epoch time `3.58316` seconds in `results.csv`. | `/root/autodl-tmp/runs/y1_yolo11m_dev_smoke_seed11171_20260714/results.csv`; run stdout captured in session note |

## Primary Results

| Metric | Candidate | Baseline | Delta | Evidence |
| --- | ---: | ---: | ---: | --- |
| Raw patient-level macro-Dice | `missing` | `missing` | `missing` | Not applicable to YOLO runtime smoke. |
| Raw ET Dice | `missing` | `missing` | `missing` | Not applicable to YOLO runtime smoke. |
| Raw TC Dice | `missing` | `missing` | `missing` | Not applicable to YOLO runtime smoke. |
| Raw WT Dice | `missing` | `missing` | `missing` | Not applicable to YOLO runtime smoke. |
| Raw HD95 (mm) | `missing` | `missing` | `missing` | Not applicable to YOLO runtime smoke. |

Paired uncertainty/statistics: `missing`

## Guardrails

| Guardrail | Result | Interpretation | Evidence |
| --- | --- | --- | --- |
| Runtime completion | `succeeded` | Model load, one train epoch, validation, and checkpoint writing completed. | `/root/autodl-tmp/runs/y1_yolo11m_dev_smoke_seed11171_20260714/train_config.json`; `/root/autodl-tmp/runs/y1_yolo11m_dev_smoke_seed11171_20260714/results.csv` |
| Local checkpoint path | `/root/autodl-tmp/checkpoints/yolo11m.pt` | Avoids slow remote YOLO checkpoint download during future runs. | `sha256sum /root/autodl-tmp/checkpoints/yolo11m.pt` |
| Complete-volume smoke validation | 310 val slices, 125 non-empty labels, 185 empty labels | Validation includes background slices, so the false-positive path is represented in smoke. | `/root/autodl-tmp/datasets/brats_yolo_dev_smoke_seed11171_20260714/dataset_manifest.json` |
| Space after run | `/root/autodl-tmp`: 150G total, 103G used, 48G available | No disk expansion required before the next Y0/Y2 step. | `df -h /root/autodl-tmp` |
| Smoke detector metric | mAP50 `0.04129`, recall `0.112` after 1 epoch | Health metric only; not used for detector selection or paper claims. | `/root/autodl-tmp/runs/y1_yolo11m_dev_smoke_seed11171_20260714/results.csv` |

Secondary raw/post analysis: `not applicable`

## Failures And Missing Evidence

- Failure stage: `not applicable`
- Error or interruption: initial remote run was interrupted because `yolo11m.pt` remote download was too slow; resumed under the same declared smoke intent using a verified local rsync checkpoint.
- Last valid evidence: completed smoke artifacts under `/root/autodl-tmp/runs/y1_yolo11m_dev_smoke_seed11171_20260714`.
- Missing/conflicting evidence:
  - No full `paper_v1` YOLO dataset yet.
  - No full Y2 training run yet.
  - No Y3 confidence scan or frozen detector operating point yet.
  - No downstream raw SAM 3D Dice or HD95 metrics.
- Conclusion limit: This report only validates runtime and data-format readiness; it cannot support detector quality, prompt-gap, or segmentation-performance claims.

## Decision And Next Action

- Decision rationale: Promote the YOLO11m AutoDL configuration through the Y1 runtime gate because the declared smoke completed with required artifacts and no disk expansion requirement.
- Next action: Materialize the full frozen `paper_v1` YOLO dataset with seed `11171` using `/root/autodl-tmp/checkpoints/yolo11m.pt` as the subsequent Y2 base checkpoint.

## Artifacts

- Manifest: `/root/autodl-tmp/datasets/brats_yolo_dev_smoke_seed11171_20260714/dataset_manifest.json`
- Metrics: `/root/autodl-tmp/runs/y1_yolo11m_dev_smoke_seed11171_20260714/results.csv`
- Logs: `missing` as standalone file; session stdout summarized in `.trellis/tasks/07-13-diagnose-prompt-and-postprocess-gap/research/y0-y1-autodl-smoke-2026-07-14.md`
- Checkpoints: `/root/autodl-tmp/checkpoints/yolo11m.pt`; `/root/autodl-tmp/runs/y1_yolo11m_dev_smoke_seed11171_20260714/weights/best.pt`; `/root/autodl-tmp/runs/y1_yolo11m_dev_smoke_seed11171_20260714/weights/last.pt`
- Predictions/plots: `missing`

## Change History

| Date | Transition | Evidence |
| --- | --- | --- |
| `2026-07-14` | `initial -> succeeded` | `/root/autodl-tmp/runs/y1_yolo11m_dev_smoke_seed11171_20260714/results.csv` |
