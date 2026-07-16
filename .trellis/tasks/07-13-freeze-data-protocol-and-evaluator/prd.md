# Freeze the BraTS Data and Evaluation Protocol

## Goal

Create the immutable `paper_v1` patient split, unify training and inference
preprocessing, and provide one native-grid 3D evaluator used by SAM, U-Net,
nnU-Net, raw predictions, and postprocessed predictions.

## Requirements

### D0: Source Audit and Split

- Use only the 1,251 complete Kaggle BRaTS 2021 Task 1 cases containing T1,
  T1ce, T2, FLAIR, and segmentation files.
- Generate patient-level train/validation/test partitions of 875/187/189 with
  seed 11171 and ratios 0.70/0.15/0.15.
- Write an immutable manifest with case IDs, source fingerprints, split seed,
  counts, and SHA-256.
- Reject missing modalities, duplicate IDs, overlap, count drift, or a source
  set that does not match the audited manifest.
- Derive the 200/43 development view only from parent train/validation with a
  separately recorded seed and manifest.

### D1: Test Lock

- Test is excluded by default from dataset preparation, cache construction,
  detector selection, threshold scans, and qualitative-case tools.
- Any command that can access test requires an explicit unlocked freeze record.
- Before unlock, tools may validate that the test partition exists but must not
  read labels or produce label-derived summaries.

### D2: Shared Preprocessing

- Load T1, T1ce, T2, and FLAIR in a single canonical channel order recorded in
  each run manifest.
- For each volume and modality, compute min/max over finite nonzero voxels and
  normalize to `[0, 1]`; keep background zero.
- Define deterministic behavior for an all-zero or constant nonzero modality.
- Use bilinear interpolation for image/probability resize and nearest-neighbour
  interpolation for labels.
- Label regions are `ET = {4}`, `TC = {1, 4}`, and `WT = {1, 2, 4}`.
- Training cache and direct inference preprocessing must agree within the
  declared float16 cache tolerance.

The cache schema, storage, and rebuild rules are defined once in the parent
task's `info.md`.

### D3: Prediction Contract

Every method exports independent binary ET, TC, and WT masks keyed by case ID
on the source native grid. Each artifact records:

- native shape, affine/spacing, and source case identity;
- model/run/checkpoint identity;
- raw or named postprocess condition;
- resize and threshold policy;
- prediction timestamp and artifact hash.

Grid mismatch fails closed. The evaluator must not compare a `256 x 256`
training-space mask directly against a native target.

### D4: Per-Case Metrics

For ET, TC, and WT, compute:

- Dice and IoU;
- HD95 in millimetres using native voxel spacing;
- sensitivity and specificity;
- predicted and ground-truth voxel counts;
- empty-region status.

Also compute `ET & ~TC`, `TC & ~WT`, total violation voxels, and whether the
case contains any hierarchy violation.

### D5: Empty-Region Semantics

- Ground truth empty and prediction empty: Dice 1, IoU 1, sensitivity
  `not_applicable`, HD95 `not_applicable`.
- Ground truth empty and prediction non-empty: Dice 0, IoU 0, sensitivity
  `not_applicable`, HD95 `not_applicable`; preserve specificity and voxel count.
- Ground truth non-empty and prediction empty: Dice 0, IoU 0, sensitivity 0,
  HD95 `not_applicable`.
- Do not replace undefined HD95 with zero or omit an empty-region case from
  class means without reporting the denominator.

### D6: Aggregate Outputs

Write one row per case to JSON and CSV, plus a machine-readable and Markdown
summary containing:

- case counts and class-specific denominators;
- mean, standard deviation, median, and interquartile range;
- empty-region counts;
- hierarchy violation voxels and violation-case rate;
- raw/postprocess condition and full run identity.

Aggregation must preserve case IDs so later paired statistical tests cannot
silently compare different populations.

## Validation Matrix

Automated tests cover:

1. Split determinism, exact counts, and pairwise disjointness.
2. Missing/duplicate cases and forbidden test access.
3. Shared normalization for normal, all-zero, constant, NaN, and Inf inputs.
4. Cached versus direct preprocessing tolerance.
5. Identical masks, disjoint masks, and known partial overlap.
6. Anisotropic-spacing HD95.
7. Every empty-region branch.
8. Hierarchy violation counting.
9. Native-grid mismatch and safe resampling/export behavior.
10. JSON/CSV aggregate agreement and stable case ordering.

## Acceptance Criteria

- [ ] `paper_v1` and development manifests are deterministic, disjoint, hashed,
      and contain the exact declared counts.
- [ ] Test access fails closed before a valid parent freeze record.
- [ ] Training, cache, YOLO preparation, and inference use one normalization
      implementation or pass an equivalence regression test.
- [ ] Every active method can export the same prediction contract.
- [ ] The common evaluator passes the validation matrix and emits complete
      per-case and aggregate artifacts.
- [ ] Raw and every postprocess condition have distinct identities and cannot
      overwrite each other.

## Out of Scope

- Model-specific checkpoint selection.
- Detector threshold selection.
- Statistical hypothesis testing and paper table formatting.
- Postprocessing algorithm design.

## Definition of Done

- Relevant unit and integration tests pass in the project verification
  environment.
- Versioned manifests and evaluator schema are archived.
- All child tasks reference the same split and evaluator identities.
