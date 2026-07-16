# Fair Medical Segmentation Baselines

## Goal

Produce a required conventional 2D U-Net result and an optional strong nnU-Net
v2 reference under the parent paper's frozen patient split and native-grid 3D
evaluator. The baselines must receive credible, predefined training protocols;
they are not intentionally weakened to favor SAM-Med2D.

## Priority

- **P0:** Conventional 2D U-Net development run, one full-data final seed,
  evaluator agreement, raw 3D result, and efficiency report.
- **P1:** nnU-Net v2 `3d_fullres`, a second U-Net seed, and additional baseline
  failure analysis.

P1 starts only after the parent task's A0-A3 development result and Plan A/B
direction are frozen.

## Shared Fairness Contract

All baselines use:

1. The exact `paper_v1` train/validation/test case partitions.
2. T1, T1ce, T2, and FLAIR inputs with the shared per-volume nonzero-minmax
   normalization unless an official baseline requires its own documented
   normalization.
3. BraTS regions `ET = {4}`, `TC = {1, 4}`, and `WT = {1, 2, 4}`.
4. Validation only for checkpoint, threshold, and training-duration choices.
5. The common native-grid patient-level evaluator and empty-region semantics.
6. Raw predictions as the headline result; parent-task postprocessing only as
   a separately labelled secondary analysis.
7. The run-manifest, seed, artifact, and test-unlock contracts from the parent
   task's `info.md`.

Fair treatment means a canonical, suitable recipe for each model and identical
data/evaluation governance. It does not require identical optimizers, epoch
counts, normalization internals, parameter counts, or compute budgets across
architecturally different methods.

## B4-U: Conventional 2D U-Net (P0)

### Input and Sampling

- Train on axial `256 x 256` four-channel slices.
- Use the shared cached per-volume normalized tensors and nearest-neighbour
  label resize.
- Retain every tumor-positive training slice.
- Add deterministic tumor-negative slices at a maximum ratio of one negative
  to three positive slices, sampled per case with the declared dataset seed.
- Retain every slice for patient-level validation and final inference.
- Apply the same spatial and intensity augmentation family used by A0-A3. Do
  not introduce U-Net-only external data or pseudo-labels.

### Architecture

Implement one standard five-level 2D U-Net:

| Component | Frozen definition |
| --- | --- |
| Input | Four MRI channels |
| Encoder widths | 32, 64, 128, 256, 512 |
| Encoder block | Two `3 x 3` convolutions, instance normalization, LeakyReLU |
| Downsampling | `2 x 2` max pooling |
| Decoder | `2 x 2` transposed convolution, skip concatenation, two-convolution block |
| Output | Three independent logits ordered ET, TC, WT |
| Parameters | Random initialization; record exact trainable/total count |

The outputs are independent sigmoid regions, not a three-class softmax,
because ET, TC, and WT are nested rather than mutually exclusive. Do not add
attention, residual blocks, pretrained encoders, deep supervision, prompt
inputs, or hierarchy projection to the raw model.

### Objective

Use an equal-weight mean over ET, TC, and WT of:

```text
loss_c = 0.5 * BCEWithLogits(logit_c, target_c)
       + 0.5 * SoftDiceLoss(sigmoid(logit_c), target_c)
loss = mean(loss_ET, loss_TC, loss_WT)
```

Soft Dice uses a documented smoothing constant and includes empty-target
slices without silently dropping them. No hierarchy loss is applied to this
baseline.

### Optimization

| Setting | Frozen definition |
| --- | --- |
| Optimizer | AdamW |
| Initial learning rate | `1e-3` |
| Weight decay | `1e-5` |
| Scheduler | ReduceLROnPlateau on validation loss, factor `0.5`, patience 5 epochs |
| Maximum epochs | 100 |
| Early stopping | 20 epochs without improvement in the 3D selection endpoint |
| Gradient clipping | Global norm 12.0 |
| AMP / batch / workers | Values selected once by the parent AutoDL runtime gate |
| Development seed | 11519 |
| Required final seed | 11519 |
| Optional second seed | 11520 |

Do not tune architecture width, loss weights, optimizer, learning rate, or
negative ratio after comparing against A0/A3. A reproducible optimization
failure permits one documented correction before the baseline is declared
failed; it does not permit a broad hyperparameter sweep.

### Checkpoint Selection

- Save immutable snapshots at least every five epochs and at every scheduler
  learning-rate change.
- Compute slice validation loss every epoch as a health metric.
- Evaluate snapshots on the complete development-validation or paper-validation
  cases at most every five epochs.
- Select by raw patient-level ET/TC/WT macro-Dice, with ET Dice as the
  co-primary guardrail, exactly as specified by the parent PRD.
- Use fixed sigmoid threshold `0.5` for the raw result. Any threshold selection
  is a secondary validation-only analysis and must not replace the raw table.

### Inference and Export

- Run the model on every native axial slice, including background-only slices.
- Resize probabilities back to the original in-plane grid with bilinear
  interpolation and threshold only after resizing.
- Export independent native-grid ET, TC, and WT masks without hierarchy repair.
- Preserve the source affine/spacing and case ID.
- Evaluate with the same common evaluator used by SAM-Med2D.

## B4-N: nnU-Net v2 `3d_fullres` (P1)

Use the official nnU-Net v2 implementation as a strong reference rather than
reimplementing its architecture.

### Dataset Conversion

- Convert only `paper_v1/train` and `paper_v1/val` cases into one nnU-Net raw
  dataset; exclude test during planning, preprocessing, and training.
- Preserve all four modalities and native geometry.
- Configure region-based training for ET, TC, and WT rather than treating the
  nested regions as mutually exclusive classes.
- Generate `splits_final.json` with exactly one fold: parent train as training
  and parent validation as validation. Do not use nnU-Net's random default
  folds for paper selection.
- Archive the source split hash, converted case list, `dataset.json`, plans,
  fingerprints, and conversion command.

### Training and Selection

- Use the official `3d_fullres` configuration, generated plans, default trainer,
  loss, optimizer, augmentation, schedule, and 1000-epoch budget.
- Do not modify topology, patch size, spacing, batch size, or training schedule
  after viewing SAM or U-Net results.
- Train fold 0 only. Do not ensemble folds for the primary reference.
- Use the official final checkpoint unless the official validation-selected
  checkpoint is predefined before test and its selection provenance is saved.
- Record nnU-Net version, plans, trainer/configuration identifiers, parameter
  count, wall-clock time, peak memory, and checkpoint hash.

If `3d_fullres` cannot finish within the P1 compute window, report the measured
resource blocker and omit nnU-Net. Do not replace it post hoc with an easier
configuration and present that as the planned strong reference.

### Export

- Export region probabilities or masks on the original case geometry.
- Convert output into independent ET, TC, and WT masks without applying the
  SAM postprocessing chain.
- Run the common project evaluator rather than copying only nnU-Net's internal
  cross-validation summary.

## Evaluator Agreement Gate

Before reporting either baseline, select at least ten validation cases covering
positive and empty ET regions. For each case:

1. Verify case ID, native shape, affine/spacing, and ET/TC/WT label mapping.
2. Score the same saved masks through the baseline export path and common
   evaluator.
3. Require identical Dice, empty-region status, voxel counts, and hierarchy
   violations; require HD95 agreement within numerical tolerance.

Any discrepancy blocks the baseline table until the conversion or evaluator
contract is corrected.

## Required Results

### Main Baseline Table

Report raw patient-level ET, TC, and WT Dice and HD95 for:

- A0 Adapter under frozen automatic prompts;
- promoted A3 or the selected Plan B intervention;
- conventional 2D U-Net;
- nnU-Net v2 when P1 completes.

Include trainable/total parameters, training wall-clock time, peak GPU memory,
and per-case inference time in a separate efficiency table. Hardware and
measurement conditions must be identical or explicitly qualified.

### Diagnostic Outputs

- Per-case JSON/CSV in the common schema.
- Predicted and ground-truth voxel counts.
- Empty-region and hierarchy-violation summaries.
- At least one predeclared failure case shared with the SAM analysis.

## Acceptance Criteria

- [ ] The conventional 2D U-Net matches the frozen architecture, loss,
      sampling, and optimization protocol.
- [ ] No test case is available to training, preprocessing decisions,
      checkpoint selection, or qualitative selection before parent unlock.
- [ ] U-Net checkpoint selection uses raw patient-level macro-Dice and the ET
      guardrail rather than slice Dice.
- [ ] Raw U-Net predictions use threshold 0.5 and no hierarchy/postprocessing
      repair.
- [ ] The ten-case evaluator agreement gate passes.
- [ ] One full-data U-Net seed produces per-case and aggregate native-grid 3D
      results plus efficiency metadata.
- [ ] nnU-Net, when run, uses the official `3d_fullres` plan and the frozen
      parent split rather than random cross-validation folds.
- [ ] Baseline failures and omissions are reported with measured reasons; they
      are not silently removed from the comparison.

## Out of Scope

- U-Net architecture or loss sweeps.
- Attention U-Net, UNet++, TransUNet, Swin-UNet, or pretrained encoders.
- nnU-Net plan modifications or fold ensembles for the primary reference.
- External data, pseudo-labels, test-time augmentation, or model ensembles.
- Baseline-specific postprocessing presented as a raw model result.

## Definition of Done

- Relevant unit, conversion, export, and evaluator-agreement tests pass.
- Required manifests, configurations, checkpoints, predictions, per-case
  metrics, aggregate tables, and efficiency measurements are archived.
- The parent task references the completed baseline run IDs and hashes.
- Any P1 omission is documented before the final paper claim is written.
