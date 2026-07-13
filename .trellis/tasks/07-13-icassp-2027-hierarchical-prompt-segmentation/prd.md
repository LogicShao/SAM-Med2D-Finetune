# ICASSP 2027: Hierarchical Prompt-Robust Brain Tumor Segmentation

## Goal

Convert the existing SAM-Med2D engineering pipeline into a reproducible
research contribution for ICASSP 2027. The paper must evaluate a training
method rather than attribute most gains to prompt heuristics and 3D
postprocessing.

## Audited Baseline Facts

- The current multi-task Adapter checkpoint was trained with independent
  class-specific ground-truth boxes. Training validation reports a slice-level
  mean Dice of 0.7560.
- Whole-volume automatic-prompt evaluation reports 0.5104 raw and 0.5460 post
  mean Dice on 167 cases. The raw-to-post gain is 0.0357.
- The 167 cases are all drawn from `data_brats_raw/val`, which is also used for
  Adapter early stopping. They are not an independent test result.
- The current manifest is a patient-level, non-overlapping split of 875 train,
  187 validation, and 189 test cases with seed 42.
- Training uses per-slice min-max intensity normalization. Whole-volume
  inference uses per-volume, nonzero-voxel min-max normalization.
- Training keeps only tumor-positive slices, while inference scans full
  volumes. Empty class masks in retained slices produce a zero-area box.
- Inference uses YOLO boxes and a WT -> TC -> ET prompt cascade. This differs
  materially from training with independent oracle boxes.
- Current 3D hierarchy postprocessing expands TC with ET and WT with TC. It
  does not improve ET directly. Morphology, connected-component filtering and
  z-axis smoothing must be evaluated separately.
- WT continuity is not a main-method candidate: on the 167-case evaluation it
  has 1,249 rescue events versus 8,128 harmful events; its mean Dice gain is
  +0.00143 with a bootstrap confidence interval crossing zero.

## Research Hypothesis

Prompt-robust training and explicit voxelwise hierarchy supervision reduce the
gap between oracle-box training and automatic-prompt whole-volume inference.
The method should improve raw patient-level 3D segmentation, not merely post-
processed output.

## Prior-Work Boundary and Submission Bar

The immediate predecessor is the IJCNN 2025 paper, formally published as
*YOLO-Driven Prompt Generation for SAM-Based Brain Tumor Segmentation*
(DOI: `10.1109/IJCNN64981.2025.11228325`). Its anonymous submission PDF is
titled *Self-Prompt Segmentation Model for Brain Tumors*. It combines a frozen
YOLO locator, three sampled point prompts, Adapter tuning, and label
erosion/dilation. Its reported experiments are a 180-image 2D slice study that
does not separately measure oracle versus automatic prompts, training/inference
prompt mismatch, hierarchy violations, or the contribution of postprocessing.

This work must therefore make a training-method claim, not repackage automatic
prompt generation:

1. PJT addresses the detected oracle-to-automatic prompt domain gap.
2. Hierarchy-aware supervision addresses the ET subset TC subset WT structure
   during training instead of repairing it only after inference.
3. The evidence is patient-level, raw 3D and paired against the current
   Adapter under an identical frozen protocol.

The target is a credible ICASSP B-tier submission, not an exhaustive medical
segmentation benchmark. The minimum defensible paper needs a clear A0-to-A3
ablation, an independent locked-test confirmation, a conventional same-protocol
baseline, and error-source/postprocessing decomposition. nnU-Net, a third seed,
and additional architectural variants improve the paper when time permits but
must not delay validating or writing the central claim.

## Method Scope

Implement and evaluate exactly these training changes first:

1. Shared preprocessing: use the same per-volume nonzero-voxel normalization
   for training and inference.
2. Prompt Jitter Training (PJT): perturb class boxes with translation and
   scale jitter; simulate missed boxes and false-positive prompts with seeded,
   configurable probabilities.
3. Hierarchy-aware supervision: add a voxelwise soft violation loss
   `mean(relu(p_et - p_tc) + relu(p_tc - p_wt))`, or an equivalent nested
   probability parameterization that guarantees `ET subset TC subset WT`.
4. Preserve the current Adapter model as the primary PEFT baseline. Do not add
   a new backbone before the above ablation is complete.

Deferred unless the primary ablation shows a clear signal:

- Scheduled-sampling prompts for the WT -> TC -> ET cascade.
- 2.5D or 3D slice-consistency training.

Out of scope for the primary method:

- Further threshold, padding, point-count, top-k, or WT-continuity grid search.
- Presenting YOLO top-k, z-prompt interpolation, web UI, or a morphology chain
  as the core research contribution.

## Data and Evaluation Protocol

1. Freeze `data_brats_paper_v1/split_manifest.json` as the only paper split.
2. Use train for fitting, val for model selection and all method development,
   and keep all 189 test cases blinded until the configuration is frozen.
3. Persist the exact case IDs, source checkpoint hashes, CLI arguments,
   environment, model parameter counts, wall-clock time and GPU memory.
4. Make all stochastic operations deterministic for each declared seed.
5. Run final primary results for two seeds at minimum. Run a third seed only
   after the two-seed result is directionally consistent and the paper schedule
   permits it.
6. Evaluate each case in 3D: ET, TC and WT Dice; HD95; sensitivity;
   specificity; hierarchy violation voxels and violation-case rate.
7. Report raw and postprocessed results separately. The main performance table
   must use raw results; postprocessing is a controlled secondary analysis.
8. Compare paired per-case Dice with a two-sided Wilcoxon signed-rank test and
   report mean +/- standard deviation across seeds.

### Full-Dataset Baseline Protocol

The Kaggle BRaTS 2021 Task 1 source is the only source for paper experiments.
It contains 1,251 complete cases and is stored on the training server at
`/opt/data/private/SAM-Med2D-Finetune/data_source/`. Each case has all five
required files: `t1`, `t1ce`, `t2`, `flair`, and `seg`.

Create a new immutable paper split named `paper_v1` from that directory with
the existing deterministic protocol:

| Split | Cases | Role |
| --- | ---: | --- |
| train | 875 | Fit SAM, YOLO, U-Net, and nnU-Net weights only |
| val | 187 | All selection, thresholds, ablations, and early stopping |
| test | 189 | Locked final evaluation only |

Use seed 42, ratios 0.70 / 0.15 / 0.15, and archive the generated manifest
hash with every run.
Do not modify an old split or reuse an untracked list. A derived development
subset may contain 200 train cases and 43 val cases only; it must be sampled
from `paper_v1/train` and `paper_v1/val` with a recorded seed. Test cases must
not appear in development paths, detector selection, threshold scans, prompt
tuning, or qualitative-case selection before the final run.

### Baseline Sequence and Gates

1. **B0: data and runtime gate.** Verify all case files, manifest disjointness,
   GPU1 visibility, environment versions, checkpoint SHA-256, model load, one
   forward/backward/optimizer step, and a 1-epoch slice-limited run. Use
   `use_amp=false`, `disable_cudnn=true`, `num_workers=2`, and GPU1 only.
2. **B1: reproducibility baseline on development data.** Reproduce the current
   multi-task Adapter with independent ground-truth class boxes. Report both
   slice-level validation metrics and patient-level 3D raw metrics. This is the
   oracle-prompt reference, not the paper headline.
3. **B2: automatic-prompt baseline.** Train the one-class YOLO locator on
   `paper_v1/train` only, select its score/NMS operating point on `paper_v1/val`,
   and run the frozen Adapter with YOLO top-1 prompts on val. Report the gap
   between oracle, jittered-oracle, and YOLO prompt conditions.
4. **B3: proposed-method development.** On the development subset, compare
   A0 current Adapter, A1 prompt jitter, A2 hierarchy loss, and A3 combined.
   Promote only a clear raw-3D validation signal to the full train/val run.
5. **B4: standardized medical baseline.** Run a conventional 2D U-Net under
   the same split and patient-level 3D evaluator. Run nnU-Net v2 only after B3
   has a clear signal and it fits the remaining schedule; it is a strengthening
   reference, not a prerequisite for the paper's central ablation.
6. **B5: final confirmation.** Run A0 and the selected proposed method for at
   least two seeds on full train/val. Freeze every setting before evaluating the
   189-case test set once per seed. Use paired case-level statistics and keep
   postprocessing as a separate secondary table.

Every run must record: split manifest SHA-256, code revision, base and
finetuned checkpoint SHA-256, seed, CLI, GPU, environment package versions,
wall-clock time, peak memory, raw metrics, and postprocessed metrics.

### Executable Baseline Plan

The dataset is now materialized on the server as symlinks, not copied data:

| Dataset root | Cases | Permitted use |
| --- | ---: | --- |
| `data_brats_paper_v1/train` | 875 | Fit model weights only |
| `data_brats_paper_v1/val` | 187 | Early stopping, threshold and method selection |
| `data_brats_paper_v1/test` | 189 | One final, frozen evaluation per final seed |
| `data_brats_paper_dev_v1/train` | 200 | Fast development only; sampled from paper train |
| `data_brats_paper_dev_v1/val` | 43 | Fast development only; sampled from paper val |

Do not create an alternative split, random slice split, or a new test subset.
The development subset is a compute-saving view of the paper split rather
than a fourth experimental population.

| Phase | Run IDs | Data | Required output | Advance only when |
| --- | --- | --- | --- | --- |
| B0 | `b0_runtime_smoke` | dev 8 train slices / 4 val slices | model load, one epoch, log, checkpoint and curves | completed on GPU1 with `use_amp=false`, `disable_cudnn=true`, `num_workers=2` |
| B0.5 | `b0_eval_contract`, `b0_throughput` | dev train/val | one 3D metrics JSON per case and measured updates/s, peak memory | all metrics and empty-mask rules are identical for SAM, U-Net and nnU-Net exports |
| B1 | `a0_dev_oracle` | 200/43 dev | current Adapter, independent GT boxes, slice metrics and raw 3D val metrics | checkpoint selection is based on raw patient-level mean Dice, not slice Dice |
| B2 | `a0_prompt_gap` | same 43 dev-val cases | no/full image, oracle, jittered oracle, YOLO top-1 raw 3D comparison | source of the oracle-to-automatic gap is quantified |
| B3 | `a1_pjt_dev`, `a2_hier_dev`, `a3_combined_dev` | same dev split | A0/A1/A2/A3 ablation with fixed seeds | A3 improves raw 3D validation mean Dice and does not materially reduce ET Dice |
| B4 | `unet2d_dev` (required), `nnunet_dev` (time permitting) | same dev split | conventional 2D U-Net raw 3D results; optional nnU-Net reference | evaluator agreement is checked on at least 10 common cases |
| B5 | final A0, final A3, U-Net; optional nnU-Net | 875/187 then 189 test | two-seed minimum final results, statistics and separate postprocess table | all configuration values are frozen before test labels are read |

`b0_runtime_smoke` passed on 2026-07-13: the one-epoch, eight-training-slice
and four-validation-slice GPU1 run created its log, checkpoint and curves.
Its validation Dice is only a pipeline-health signal because the sample is too
small and not patient-level 3D evaluation.

`b0_eval_contract` and `b0_throughput` passed on 2026-07-13. The common
`brats_metrics.py` evaluator now emits native-grid Dice, IoU, HD95 in mm,
sensitivity, specificity, voxel counts, empty-region status, and hierarchy
violations. Its unit tests cover identical masks, anisotropic HD95, empty-mask
semantics, violations, and grid mismatch. A one-case native-grid inference
smoke wrote complete JSON, CSV and Markdown results under
`workdir_eval_contract/b0_eval_contract_native_grid` without using test data.
The smoke checkpoint is intentionally not a performance result.

The 100-update GPU1 benchmark used the full 200-case development training
view (13,068 positive slices), batch size 1, two workers, AMP disabled and
cuDNN disabled. The training segment took 25.4722 seconds (3.9259 updates/s)
with 4,908.7 MiB peak allocated and 5,176.0 MiB peak reserved CUDA memory.
The complete record is in
`workdir_benchmark_gpu1/logs/b0_throughput_100steps_adapter/metrics.csv`.
This is the initial capacity estimate, not a final-training time claim.

### B1 Training-Input Decision

Before B1, implement a measured input-pipeline change rather than reducing
the frozen case split. The current dataset opens five compressed NIfTI files
for every sampled slice, so it must be profiled before assuming the GPU is the
sole bottleneck.

1. Build a versioned, per-case on-disk cache of per-volume nonzero-voxel
   normalized four-modality tensors and segmentation labels. Keep online
   spatial/intensity augmentation after cache loading.
2. Add deterministic negative-slice sampling. Retain every tumor-positive
   slice and support one tumor-negative slice for every three positive slices.
   This is B0 input-pipeline infrastructure and the prescribed PJT setting,
   not the A0 baseline setting. The ratio is an explicit CLI argument, not an
   implicit dataset change.
3. Profile each training epoch with mean data wait, mean GPU compute time,
   CUDA peak memory, and optional sampled `nvidia-smi` GPU utilization. Run
   the same short benchmark with the cache disabled and enabled before
   choosing worker count or batch size.
4. Do not reduce the 875-case final training split. The 200/43 development
   view remains the only compute-saving dataset view for method selection.

Implementation and profile result (2026-07-13): the full 200/43 development
cache contains 243 cases and occupies approximately 19 GB. The cache uses
four-channel per-volume nonzero-minmax `float16` tensors and `uint8`
segmentation labels, served through memory-mapped files. With otherwise equal
GPU1 settings, cache reduced mean batch wait from 29.4 ms to 6.7 ms and
increased batch-1 throughput from 4.111 to 4.458 updates/s. Batch 4 achieved
14.018 samples/s and 52.3% sampled GPU utilization; batch 8 only reached
16.041 samples/s while halving optimizer updates, so batch 4 is the current
development default. The 1:3 positive-to-negative configuration produced
13,068 positive and 4,353 negative slices, remained stable for 100 steps at
batch 4, and reached 14.610 samples/s with 55.1% sampled GPU utilization.
These are capacity measurements only, not segmentation results.

#### Strategic Review Before B1

The performance investigation is complete. Do not spend more paper time on
batch 8, worker-count tuning, cache size tuning, AMP, cuDNN, or DDP unless the
measured batch-4 throughput regresses materially. Batch 4, two workers and an
eight-case mmap LRU are the frozen development defaults.

The negative-slice implementation is validated infrastructure, but it must not
be enabled in A0. A random box on an empty slice is false-positive prompt
simulation and therefore belongs to PJT. Enabling it in A0 would contaminate
the A0/A1 ablation. Use these definitions:

| Run | Negative ratio | Negative prompt | Positive-box perturbation | Hierarchy loss |
| --- | ---: | --- | --- | --- |
| A0 | 0 | zero | none | none |
| A1 | 1/3 | random | PJT enabled | none |
| A2 | 0 | zero | none | enabled |
| A3 | 1/3 | random | PJT enabled | enabled |

This table overrides the earlier B0 pipeline-capacity suggestion for every
A0--A3 paper result. Thus the completed A0 run uses all 13,068 positive
development-training slices, no sampled empty slices and only independent
oracle class boxes. Negative slices are deliberately introduced first in A1
and A3, together with positive-box jitter, so their effect is attributable to
PJT rather than hidden inside the baseline.

Before launching A0, close three short reproducibility gaps:

1. Add an explicit global training seed and seed Python, NumPy, Torch,
   DataLoader shuffling/workers and augmentation randomness. `dataset_seed`
   currently controls only negative-slice selection and negative boxes.
2. Save immutable epoch snapshots. The current script only overwrites
   `best_model.pth` according to slice-level validation Dice, so it cannot
   support retrospective raw patient-level selection.
3. Add a cache-contract test comparing cache tensors with the inference
   per-volume nonzero-minmax path within the declared float16 tolerance, and
   reject incompatible cache schema/dtype/shape metadata.

Run A0 once for five epochs and save epochs 1, 3 and 5 from that single
trajectory; do not launch three independent training jobs. Evaluate all three
snapshots on all 43 development-validation cases with `upper_bound` prompts,
postprocessing and HTML rendering disabled, and select by raw patient-level
mean Dice with ET Dice as the guardrail. Slice-level validation is a health
metric only.

If epoch 5 is still more than 0.5 Dice point above epoch 3 and the trajectory is
monotonically improving, extend the same run policy to at most epoch 8 or 10.
Otherwise freeze the best of epochs 1, 3 and 5. Do not tune learning rate,
negative ratio or augmentation during B1.

For the later A0/A1/A2/A3 headline comparison, every method must use the same
checkpoint-selection prompt condition and the same 43 validation cases. Do not
select A0 with oracle prompts while selecting A3 with YOLO prompts in the same
comparison table. Prefer frozen YOLO top-1 raw 3D Dice for the automatic-prompt
headline after the B2 prompt-gap diagnostic is complete.

The B1 launch gate passed on 2026-07-13 after server-side compilation, 12 unit
tests and a GPU1 cached backward smoke. The smoke used the A0 input definition
and wrote an immutable epoch snapshot while counting actual batch samples in
the runtime CSV. It is a health check only, not a segmentation result.

#### Evaluation Contract Before Any Baseline Claim

Extend the common 3D evaluator before B1 beyond its current Dice/IoU output.
For each ET, TC and WT mask it must produce Dice, HD95 in millimetres using
the NIfTI affine/spacing, sensitivity, specificity, predicted/ground-truth
voxel counts, and hierarchy violations (`ET & ~TC`, `TC & ~WT`). Preserve a
machine-readable per-case CSV/JSON and a run-level summary. The evaluator
must explicitly record these conventions:

1. Resample each prediction to the original ground-truth grid before scoring;
   do not score 256 x 256 training-space masks against native volumes.
2. For an empty ground-truth region, score Dice and HD95 according to a single
   documented BraTS-compatible rule and additionally report the number of such
   cases. Never hide them by omitting cases from a class mean.
3. Compute HD95 only with the native voxel spacing; report `not_applicable` for
   a mathematically undefined empty-surface pair rather than substituting a
   zero distance.
4. Primary selection metric is unpostprocessed patient-level mean Dice over
   ET/TC/WT. Per-class Dice, particularly ET, are co-primary guardrails.
5. Thresholds, morphology parameters and detector operating points are chosen
   only on `paper_v1/val`, then frozen and replayed on test unchanged.

#### Compute-Aware Training Policy

The GPU1 smoke establishes correctness but not the duration of a full run.
Before scheduling any long baseline, run a fixed 100-update benchmark on the
200-case development split with the exact image size, batch size and workers.
Record updates/s, peak memory and positive-slice count. Use that measurement
to set a wall-clock budget and epoch/step count; do not assume that 300 epochs
is feasible merely because it was used by an earlier YOLO experiment.

Long jobs run serially on GPU1. GPU0 may validate evaluator outputs on a small
case set only. The order of expensive work is A0, prompt-gap diagnosis,
PJT/hierarchy ablation, 2D U-Net, then optional nnU-Net. nnU-Net is a strong
reference, not a gate for either the proposed-method ablation or the submission
minimum; this preserves time for the paper's central prompt-robustness question
if its full 3D training proves slow on an 11 GB GPU.

## Compute Environment Constraint

The available server has two RTX 2080 Ti GPUs, but its verified DDP gate
conclusion is negative for long real-model runs. GPU0 is unstable for long
training, AMP hangs, cuDNN is unstable in the existing YOLO environment, and
the two GPUs have no peer access. Do not use DDP for paper experiments.

Formal training policy:

1. Use physical GPU1 only for all long SAM and YOLO training.
2. Reserve GPU0 for short smoke tests or isolated evaluation only. Do not
   report throughput collected while another GPU job is running.
3. Build an isolated SAM environment; do not reuse the verified system Python
   3.8 / Torch 1.14 YOLO environment. This repository imports `torch.amp` and
   requires modern `numpy`, `scipy`, `pandas`, and `peft` versions that are not
   compatible with that environment.
4. Before a long run, pass a GPU1-only single-card gate: import, forward,
   backward, optimizer step, 3 epochs, then 50 epochs using the exact planned
   batch size and data-loader configuration.
5. Make AMP, cuDNN policy, worker count and seeds explicit CLI options and log
   them in each run. The current multi-task script defaults AMP to true and
   hard-codes `num_workers=4`; it also does not consume `J3S2_DISABLE_CUDNN`.
   These must be fixed and smoke-tested before relying on the server policy.
6. Use `CUDA_LAUNCH_BLOCKING=1` only for diagnosis unless a GPU1 stability
   gate demonstrates it is required. It serializes CUDA launches and should
   not be assumed necessary for final throughput measurements.

YOLO is allowed only as an automatic-prompt provider, trained solely on the
frozen train split. Its model-selection data must be validation only; test
labels may never be used for detector selection. Retain YOLO only if the
oracle-box / jittered-box / YOLO-box diagnostic shows that it supports a
credible automatic-prompt result. It is not a primary paper contribution.

## Required Experimental Matrix

### Diagnostic prompt decomposition

Use the same frozen checkpoint and the same cases:

| Condition | Purpose |
| --- | --- |
| Full-image or no-prompt baseline | Quantify intrinsic model behavior |
| Oracle class boxes | Training-compatible prompt upper bound |
| Jittered oracle boxes | Isolate localization robustness |
| YOLO top-1 box | Automatic end-to-end baseline |
| YOLO box plus class-specific prompts | Current pipeline behavior |

### Training ablation

| ID | Shared normalization | PJT | Hierarchy loss | Purpose |
| --- | --- | --- | --- | --- |
| A0 | Yes | No | No | Reproduced Adapter baseline |
| A1 | Yes | Yes | No | Prompt robustness effect |
| A2 | Yes | No | Yes | Hierarchy effect |
| A3 | Yes | Yes | Yes | Proposed method |

### Postprocessing ablation

For A0 and A3 report:

1. Raw predictions.
2. Hierarchy projection only.
3. Morphology and component filtering only.
4. Full existing postprocessing.

## Comparator Requirements

Run all methods on the same frozen data protocol and report identical 3D
metrics:

1. A conventional 2D U-Net as the required same-protocol reference.
2. nnU-Net v2 as a preferred strong 3D reference when it can finish within the
   paper schedule.
3. SAM-Med2D frozen and Adapter. LoRA is supplementary only and must not delay
   the A0-to-A3 comparison.

If the proposed method remains substantially behind nnU-Net, position the
paper as parameter-efficient automatic-prompt robustness rather than a
state-of-the-art tumor segmentation claim.

## Research Reference

- [`research/ijcnn2025-predecessor-gap.md`](research/ijcnn2025-predecessor-gap.md)
  records the local-PDF audit, the non-overlapping claim boundary, and the
  minimum evidence required for this submission.
- [`research/b0-cache-b1-strategy-review.md`](research/b0-cache-b1-strategy-review.md)
  records the performance-pipeline review, A0 contamination risk, and the
  exact B1 execution gate.

## Acceptance Criteria

- No patient overlap across train, val and test; manifests are committed.
- Train/inference normalization behavior is shared and unit-tested.
- The new training path has deterministic, configurable jitter and hierarchy
  coefficients, and preserves the old baseline through explicit flags.
- A3 shows at least a 1.5-point raw mean-Dice improvement over A0 on the fixed
  development validation set, with no ET Dice decrease above 1 point; otherwise
  stop method expansion and revise the claim before test.
- The final test result has two-seed minimum metrics, paired significance
  tests, the required U-Net comparison, prompt and postprocess ablations, and
  at least three representative failure cases. Include nnU-Net and a third seed
  when the central result is already complete and the schedule allows.

## 60-Day Milestones

| Days | Deliverable |
| --- | --- |
| 1-7 | Freeze protocol, normalize preprocessing, seed control, unified 3D evaluator, reproduce A0 |
| 8-14 | Prompt diagnostic table and raw/postprocess decomposition |
| 15-28 | Implement PJT and hierarchy loss; select A0-A3 on validation |
| 29-38 | Run conventional U-Net; add nnU-Net only if the core result is frozen |
| 39-50 | Freeze configuration; run two seeds and the untouched test set; add a third seed if feasible |
| 51-60 | Statistics, figures, failure analysis, paper and reproducibility package |

## Risks and Decision Gates

- Do not access test metrics before the A0-A3 design and hyperparameters are
  frozen in the task record.
- If prompt decomposition shows that YOLO localization dominates the error,
  report it and prioritize PJT; do not conceal it with postprocessing.
- If A3 has no validation signal, stop extending the method and reframe the
  paper around the measured prompt-robustness analysis.
- If A3 fails the 1.5-point / ET guardrail on the fixed development validation
  set, do not access test labels or add new method components. Diagnose the
  prompt gap and either revise the method once or stop the paper track.
- Do not use the prior `fixed20` or 167-case validation result as a final
  paper headline number.
