# Freeze YOLO Prompts and Diagnose the Automatic-Prompt Gap

## Goal

Create one reproducible YOLO11m tumor locator under the frozen `paper_v1`
patient split, freeze its predictions, and quantify how much A0 segmentation
quality is lost between oracle, jittered-oracle and automatic YOLO prompts.
This task supplies a controlled prompt condition for later SAM ablations; it
does not attempt to make YOLO a second paper contribution.

## Current State (2026-07-14)

- Y0/Y1 smoke is complete on 2 train and 2 validation cases. The real-data
  export, one-epoch YOLO11m train/validation path, checkpoint writing and
  AutoDL memory gate passed.
- Smoke `mAP50=0.04129` and recall `0.112` are runtime-health observations
  only. They do not measure the formal detector or support a quality claim.
- The next execution step is full `paper_v1` materialization, followed by the
  single Y2 training run after its pre-run manifest is written.

## Retraining Execution Update (2026-07-16)

The first formal YOLO11m detector converged but failed the Y3 Stage A gate.
The permitted recall-oriented retraining path is now automated by
`sam_med2d_finetune.tools.run_yolo_retrain_pipeline` instead of ad hoc shell
steps.

Fixed retraining defaults:

- model `YOLO11m`, optimizer `SGD`, `lr0=0.01`, momentum `0.9`,
  `warmup_bias_lr=0.0`, AMP enabled, workers `2`, seed `11171`, deterministic
  training enabled.
- MRI-safe HSV remains `h=0,s=0,v=0.1` by relying on the existing YOLO dataset
  preprocessing and not introducing color-channel augmentation changes in this
  task.
- Effective batch target is `64`: image size 512 uses hardware batch `32`;
  the 640 fallback uses hardware batch `16`.
- Screen runs train for 15 epochs on one third of training slices, then run the
  complete 187-case Y3 scan over `best.pt` and `last.pt`.
- A screen passes only with zero fully missed cases, coverage@0.50 at least
  `0.98`, and maximum consecutive coverage misses no greater than `2`.
- The screen order is S1 `512/mosaic1.0/scale0.5/box7.5`, S2
  `512/mosaic0/scale0.2/box7.5`, and S3
  `512/mosaic0/scale0.2/box10`. Passing any 512 screen stops the remaining
  screens and skips 640.
- Only if all three 512 screens fail, the best 512 configuration is retried at
  image size 640 and hardware batch `16`.
- A passing screen selects configuration only. The formal run always restarts
  from the original YOLO11m checkpoint with all training data, 100 epochs and
  patience 20; screen checkpoints are never frozen.
- After formal training, unchanged Y3 scans both `best.pt` and `last.pt`.
  Passing formal Y3 terminates the YOLO path and hands off to A0. Failing
  formal Y3 records a complete terminal rejection and enters P2 without a
  second formal training attempt.

The pipeline writes `pipeline_manifest.json`, `REMOTE_PIPELINE_COMPLETE`,
`READY_TO_POWER_OFF`, `PIPELINE_FAILED`, `SHUTDOWN_REQUESTED`, and
`SHUTDOWN_FAILED` markers. `REMOTE_PIPELINE_COMPLETE` means remote training/Y3
artifacts are complete. When explicitly launched with `--shutdown_on_exit`, all
terminal states request AutoDL shutdown through the official fixed
`/usr/bin/shutdown` command after evidence is flushed to disk. Do not probe that
command with `--help`, `--version`, `test`, `ls`, or `sha256sum`; trust the
AutoDL documentation and call it only as the final terminal action.
`READY_TO_POWER_OFF` remains a post-hoc local reporting/synchronization marker
for the next boot and no longer blocks automatic shutdown.

## Requirements

### Y0: Dataset Protocol

- Input cases come only from `data_brats_paper_v1/train` and
  `data_brats_paper_v1/val`.
- The dataset-preparation interface requires explicit splits. Test is excluded
  by default and cannot be materialized during Y0-Y3.
- Produce one-class YOLO labels named `Tumor` from the WT bounding box
  (`seg > 0`) on each positive axial slice. Expand each side by a fixed 10% of
  box width/height and clip to the image; do not tune this padding.
- Build pseudo-RGB images in `T1ce/T2/FLAIR` order with shared per-volume
  nonzero min-max normalization.
- Training contains all positive slices plus deterministic per-case negative
  sampling capped at one negative per three positives, seed 11171.
- Validation contains every slice from all 187 validation cases so detector
  false positives are measured across complete volumes.
- Write `dataset_manifest.json` with split hash, case and slice IDs, counts,
  class map, modalities, normalization, box definition and seed.
- Reject duplicate case/slice IDs, train/val overlap, missing source files and
  any test case before writing training outputs.

### Y1: Runtime Gate

- Use the verified AutoDL `sam-med2d` environment and `cuda:0` on the single
  RTX 4090D.
- Run dataset integrity checks, model load, one forward pass, one backward
  update and a short 1-epoch smoke.
- Start with `imgsz=320`, `batch=16`, `workers=4`, `seed=11171`, and `amp=false`.
- Reduce batch to 8 only after a reproducible AutoDL CUDA OOM; do not change
  the model size as the first response. Promote AMP or another worker count
  only after the parent task's fixed AutoDL profile gate.

### Y2: Detector Training

- Run ID: `y0_yolo11m_paper_v1_seed11171`.
- Initialize from `yolo11m.pt`.
- Train for at most 100 epochs with patience 20 and save every 10 epochs.
- Refuse a non-empty run directory under the same run ID unless this is a
  documented resume. Write `manifest.json` before importing or starting the
  Ultralytics trainer.
- The pre-run manifest records code revision and dirty state, dataset/data-YAML
  hashes and counts, base-checkpoint hash, full resolved configuration, seed,
  deterministic flag, environment packages, GPU and declared artifact paths.
- On success, failure or interruption, append exit status, end time, wall-clock
  time, peak allocated/reserved GPU memory, selected checkpoint and all
  available checkpoint hashes. A failed run is evidence and must not disappear.
- Do not sweep model size, optimizer, augmentation family or image size.
  `yolov8m.pt` is a fallback only after a documented YOLO11m AutoDL
  compatibility, memory or convergence failure.
- Persist `best.pt`, `last.pt`, resolved data YAML, train config, Ultralytics
  version, environment packages, GPU, wall-clock time and checkpoint hashes.

### Y3: Detector Selection

- Evaluate `best.pt` and `last.pt` on every validation slice.
- Scan confidence `0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.10` with NMS IoU
  fixed at `0.60` using top-1 output.
- Define coverage as `intersection(predicted box, GT box) / GT-box area`.
  Report positive-slice coverage recall at 0.50 and 0.80, any-box recall,
  fully missed cases, maximum consecutive misses in each tumor z-span,
  background false-positive slice rate, boxes per positive/negative slice and
  predicted/GT box-area ratio. IoU and mAP are supplementary.
- A fully missed case has no tumor-positive slice with coverage at least 0.50.
  A missed tumor-positive slice also means coverage below 0.50; consecutive
  misses are counted only across adjacent axial indices inside each case.
- Stage A is detector-only and uses all 187 validation cases. Rank operating
  points lexicographically by fully missed cases, missed positive slices,
  maximum consecutive misses, coverage recall, background false positives and
  box-area ratio. A larger covering box is preferred to a tighter box that
  misses the target. Keep at most two checkpoint/operating-point combinations.
- Stage B evaluates only those shortlisted candidates with the frozen A0 on the
  fixed 43-case dev-val view. Select one by raw patient-level A0 macro-Dice;
  raw ET Dice and background false positives are guardrails. Downstream Dice
  must not be used to alter the Stage A shortlist after it is formed.
- Freeze checkpoint SHA-256, confidence, NMS IoU, `max_det=1`, `topk=1`,
  `prompt_box_strategy=top1`, image size and preprocessing.
- Export one versioned prediction JSON for every scanned operating point,
  keyed by case ID and slice ID and containing top-1 normalized boxes and
  confidence. Bind every export to checkpoint and dataset-manifest SHA-256;
  validate case/slice completeness before Stage B. After Stage B, retain the
  chosen export as the frozen automatic-prompt artifact.
- Require zero fully missed validation cases, coverage recall at 0.50 >= 0.98,
  and no sequence longer than two consecutive missed tumor-positive slices.
  If no candidate passes, permit one documented recall-oriented retraining
  change before reconsidering the detector path.

### Y4: Prompt-Gap Decomposition

Use the same frozen A0 checkpoint and the same 43 cases for every condition:

| Condition | Purpose |
| --- | --- |
| Full-image box | No-localization reference |
| Oracle ET/TC/WT boxes | Training-compatible upper bound |
| Jittered oracle boxes | Controlled localization-error curve |
| Frozen YOLO top-1 box | Automatic end-to-end baseline |
| Frozen YOLO plus existing class prompts | Secondary current-pipeline diagnostic |

- Headline metrics are raw patient-level 3D ET/TC/WT Dice, HD95 and hierarchy
  violations. Postprocessed output is a separate secondary table.
- Save one machine-readable row per case and condition. Use paired case IDs in
  every comparison.
- Persist YOLO predictions by case ID and slice ID. All later SAM methods and
  seeds replay these files instead of rerunning the detector.
- Measure YOLO-versus-oracle box translation, scale and miss distributions.
  These measurements may define PJT ranges, but no online or joint YOLO/SAM
  training is introduced.

## Acceptance Criteria

- [ ] YOLO train/val case IDs exactly match the corresponding `paper_v1`
      manifest partitions and contain no test case.
- [ ] Dataset preparation is deterministic and records all positive/background
      slice counts and source hashes.
- [ ] Pseudo-RGB preprocessing and box coordinates match inference within a
      regression-test tolerance.
- [ ] An AutoDL `cuda:0` smoke and the single full YOLO11m run complete with archived
      configuration and checkpoint hashes.
- [ ] The Y2 run writes `manifest.json` before training and records a terminal
      success, failure or interruption state with exit status, runtime, peak
      memory and artifact hashes.
- [ ] Detector selection scans complete validation volumes, including all
      background slices, applies the recall gate, and freezes one top-1
      operating point.
- [ ] Every shortlisted operating point has a complete case/slice prediction
      export bound to detector and dataset-manifest hashes; the final export is
      replayed without invoking YOLO online.
- [ ] No validation case is fully missed; top-1 GT-box coverage recall at 0.50
      is at least 0.98 and no case contains more than two consecutive missed
      tumor-positive slices.
- [ ] Frozen YOLO prompts beat the full-image/no-localization A0 condition on
      raw 3D mean Dice; otherwise the failure and one diagnosis cycle are
      recorded before this paper path is reconsidered.
- [ ] The 43-case prompt-gap table uses identical A0 weights and case IDs for
      every prompt condition.
- [ ] Frozen prediction files are replayable and are the only automatic prompts
      used by A0/A1/A2/A3 and final SAM seeds.
- [ ] Test labels and test-derived detector metrics remain untouched until all
      detector and SAM settings are frozen.

## Definition of Done

- Relevant unit/integration tests pass.
- Dataset manifest, run configuration, detector hashes and prompt predictions
  are archived under versioned run directories.
- Raw per-case and aggregate prompt-gap outputs are generated as JSON/CSV/MD.
- The main ICASSP PRD records the frozen detector ID and operating point.

## Decision (ADR-lite)

**Context**: Retraining YOLO per SAM method or seed would change the prompt
distribution and confound the PJT/hierarchy ablation. Positive-only detector
validation would also hide full-volume false positives.

**Decision**: Train one single-class WT locator, validate it on complete
volumes, freeze one top-1 operating point, persist its predictions, and reuse
those predictions for every SAM experiment.

**Consequences**: The design sacrifices detector-variance estimates and YOLO
architecture sweeps, but sharply reduces compute and gives the SAM comparison
a stable causal interpretation suitable for the submission target.

## Out of Scope

- Joint or end-to-end YOLO/SAM training.
- Separate YOLO detectors for ET, TC and WT.
- YOLO model-size, optimizer or augmentation sweeps.
- Retraining YOLO for each SAM method or random seed.
- Top-k, temporal interpolation or continuity heuristics as the primary method.
- Any test-set threshold, checkpoint or qualitative-case selection.

## Technical Notes

- `src/sam_med2d_finetune/tools/prepare_yolo_data.py` creates WT boxes and
  `T1ce/T2/FLAIR` PNGs with explicit train/val selection, deterministic
  per-case negative sampling and `dataset_manifest.json` provenance.
- `src/sam_med2d_finetune/tools/train_yolo.py` writes a pre-run manifest and
  terminal provenance; paper defaults and the immutable run ID must still be
  passed explicitly.
- `src/sam_med2d_finetune/tools/evaluate_yolo_recall.py` implements the fixed
  confidence scan, coverage/case-miss metrics, two-candidate shortlist and
  hashed per-slice prediction exports. Formal use requires the complete val
  view and `max_det=1`.
- `src/sam_med2d_finetune/tools/run_yolo_retrain_pipeline.py` orchestrates the
  recall-oriented screen sequence, one formal retrain, automatic Y3 scans,
  terminal marker ordering and readiness checks. It may call only the official
  AutoDL `/usr/bin/shutdown` command as the final action when launched with
  `--shutdown_on_exit`; it must not call `poweroff`, systemd shutdown, or kill
  PID 1.
- `src/sam_med2d_finetune/inference/volume.py` supports online `yolo_box` diagnostics
  and strict `frozen_yolo_box` replay through `--yolo_predictions`. Frozen replay
  validates schema, hashes, case/slice completeness and normalized top-1 boxes,
  then shares the same persisted tumor box across ET/TC/WT without importing
  Ultralytics or loading a detector checkpoint.

## Research Reference

- [`research/yolo-protocol-audit.md`](research/yolo-protocol-audit.md) records
  the existing tool capabilities, protocol gaps and recall-first decision.
