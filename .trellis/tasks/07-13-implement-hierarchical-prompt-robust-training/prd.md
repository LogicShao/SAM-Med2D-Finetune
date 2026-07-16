# Implement Hierarchical Prompt-Robust Training

## Goal

Implement shared preprocessing, Prompt Jitter Training (PJT), and voxelwise
hierarchy supervision while preserving an explicit A0 Adapter baseline. Run the
fixed A0/A1/A2/A3 development ablation required by the parent research task.

## Fixed Ablation

| Run | Negative ratio | Negative prompt | Positive-box jitter | Hierarchy loss |
| --- | ---: | --- | --- | --- |
| A0 | 0 | zero | none | none |
| A1 | 1 negative / 3 positives | seeded random box | enabled | none |
| A2 | 0 | zero | none | enabled |
| A3 | 1 negative / 3 positives | seeded random box | enabled | enabled |

Shared per-volume nonzero-minmax normalization is enabled for all four runs and
is not counted as an A3 method contribution.

## Requirements

### M0: Reproducible Baseline

- Preserve the current Adapter-tuned SAM-Med2D backbone and trainable-module
  policy behind explicit configuration.
- Seed Python, NumPy, Torch, DataLoader workers/generator, dataset sampling,
  augmentation, negative boxes, and PJT.
- Save immutable epoch snapshots and complete run manifests.
- A0 contains only tumor-positive slices and independent oracle class boxes.
  Negative prompts must not be hidden inside the baseline.
- Existing behavior remains accessible through explicit flags for regression
  testing, but historical preprocessing mismatch is not used for paper A0.

### M1: Prompt Jitter Training

For a positive class box, independently support:

- translation of centre coordinates;
- width/height scale perturbation;
- clipping to image bounds with a valid minimum extent;
- missed-box simulation;
- seeded false-positive boxes on sampled negative slices.

PJT ranges are derived once from the frozen YOLO-versus-oracle translation,
scale, and miss distributions on validation. Record the derivation, selected
quantiles, and final values before A1/A3 training. Do not call YOLO online or
jointly train the detector.

The same `(case_id, slice_id, class, epoch, seed)` key must reproduce the same
prompt independent of DataLoader worker scheduling.

### M2: Hierarchy Supervision

Add the voxelwise soft violation term:

```text
L_hier = mean(relu(p_et - p_tc) + relu(p_tc - p_wt))
L_total = L_seg + lambda_hier * L_hier
```

where probabilities are aligned on the same spatial grid. `lambda_hier=0`
must reproduce the non-hierarchy code path within numerical tolerance.

Choose `lambda_hier` from a predeclared small validation-only set before A2/A3
headline comparison, record every attempted value, and freeze one value using
the parent raw macro-Dice endpoint plus ET guardrail and hierarchy diagnostics.
Do not select it using postprocessed output or test data.

An equivalent nested probability parameterization is allowed only if selected
before implementation and documented as a parent-PRD revision; do not mix both
mechanisms in the same ablation.

### M3: Training Budget and Selection

On the 200/43 development view:

1. Run A0 once for five epochs and keep only `best_model.pth` by default.
   Do not pass `--save_epochs` in standard A0/A1/A2/A3 launches.
2. Record the best checkpoint epoch from `metrics.csv`, the resolved training
   config, and the checkpoint SHA-256. The current best checkpoint is selected
   by slice-level validation Dice and must be labelled as a development proxy,
   not a formal paper checkpoint-selection endpoint.
3. Evaluate the retained A0 `best_model.pth` on all 43 cases with the frozen
   automatic-prompt files, raw output, and HTML/postprocessing disabled.
4. If the retained checkpoint is still clearly undertrained, extend the shared
   epoch budget to at most epoch 10 and still retain only the resulting
   `best_model.pth`; otherwise freeze the five-epoch budget.
5. Apply the same epoch budget and storage policy to A1/A2/A3.
6. Select every method with the same checkpoint proxy and then report raw
   patient-level macro-Dice and the ET guardrail under identical prompt
   conditions and cases.

Slice Dice and training loss are health metrics only. Do not retune learning
rate, augmentation, batch, negative ratio, or training duration separately per
ablation arm after results are visible.

### M4: Headline Evaluation

- Replay the same frozen YOLO prediction files for A0/A1/A2/A3.
- Report raw ET/TC/WT Dice, HD95, and hierarchy violations per case.
- Run parent-defined postprocessing decomposition only after raw predictions
  are archived.
- Use seed 11171 for the required development ablation.
- Promote only A0 and A3 to the two required full-data seeds when A3 passes the
  parent G2 gate. A1/A2 multi-seed runs are P1.

## Decision Rules

- A1 versus A0 estimates prompt-robustness contribution.
- A2 versus A0 estimates hierarchy-supervision contribution.
- A3 versus A0 estimates the combined method.
- Candidate promotion follows the parent PRD's frozen Fallback A-D gates:
  select the simplest strongest candidate among A1/A2/A3 by raw patient-level
  macro-Dice and the predeclared fallback endpoints; the combined A3 is not
  automatically preferred over a stronger simpler A1 or A2. A 1.5-point fixed
  threshold is superseded by the parent's calibrated Fallback A margin.
- Permit one diagnosis/revision cycle before G3. Further method expansion or
  test access is forbidden after a second failure.

## Test Matrix

Automated tests cover:

1. A0 flags reproduce baseline prompts and loss.
2. PJT is deterministic across workers and seeds vary its output.
3. Jittered boxes remain valid and clipped at image boundaries.
4. Missed and false-positive prompt frequencies match configured probabilities.
5. Hierarchy loss is zero for nested probabilities and positive for violations.
6. Gradients from hierarchy loss reach the intended trainable parameters.
7. `lambda_hier=0` matches the base segmentation loss.
8. A0/A2 reject negative-slice contamination.
9. Frozen prompt replay covers every requested case/slice.
10. Retained checkpoint and run-manifest identities are immutable.

## Acceptance Criteria

- [ ] A0/A1/A2/A3 are explicit configurations with no hidden input changes.
- [ ] PJT parameters are derived and frozen from validation detector errors.
- [ ] Hierarchy supervision is configurable, tested, and reported separately.
- [ ] The four development arms share data, budget, prompt files, evaluator,
      seed policy, and checkpoint-selection endpoint.
- [ ] Raw per-case predictions and metrics are archived before postprocessing.
- [ ] Parent G2 is applied exactly once after the allowed diagnosis cycle.

## Out of Scope

- New backbones, LoRA sweeps, 2.5D/3D consistency, and cascade scheduled
  sampling on the P0 path.
- Online or joint YOLO/SAM training.
- Detector, threshold, padding, top-k, or postprocess tuning as method rescue.

## Definition of Done

- Unit/integration tests and AutoDL runtime gates pass.
- A0/A1/A2/A3 development artifacts and the signed G2 decision are archived.
- The parent task records whether Plan A or Plan B is selected.
