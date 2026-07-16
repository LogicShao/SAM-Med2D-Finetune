# ICASSP 2027 Shared Technical Design

## Purpose

This document defines execution mechanics shared by the parent research task
and its child tasks. The parent `prd.md` owns research claims, priorities,
metrics, and decision gates. Each child PRD owns method-specific requirements.

Do not duplicate method hyperparameters here. When a child PRD conflicts with
this document on shared execution mechanics, stop the run and resolve the
conflict before producing evidence.

## Verified AutoDL Environment

All new paper experiments run on the AutoDL instance. Measurements and
checkpoints from the retired group server are historical artifacts only.

| Item | Contract |
| --- | --- |
| Compute | One RTX 4090D with 24 GiB VRAM, exposed as `cuda:0` |
| CPU / RAM | 15 CPU cores / 80 GiB RAM |
| Data disk | `/root/autodl-tmp` |
| Repository | `/root/SAM-Med2D-Finetune` |
| Conda environment | `/root/autodl-tmp/envs/sam-med2d` |
| Python / PyTorch / CUDA | Python 3.11 / PyTorch 2.6.0+cu124 / CUDA 12.4 |
| Base checkpoint | `/root/autodl-tmp/checkpoints/sam-med2d_b.pth` |
| Run root | `/root/autodl-tmp/runs` |

The detailed environment audit is stored in
`research/autodl-environment-validation-2026-07-14.md`.

## Storage Contract

- Keep source code in the repository on the system disk.
- Keep raw data, immutable splits, caches, checkpoints, logs, predictions, and
  results below `/root/autodl-tmp`.
- Do not copy large generated artifacts into Git.
- Expand the AutoDL data disk before a full-cache run if measured capacity is
  insufficient; do not silently switch storage roots during an experiment.
- A run artifact must not depend on an unrecorded symlink target or temporary
  system path.

## Run Identity

Every evidence-producing run has a unique immutable run ID. Use this pattern:

```text
<stage>_<method>_<data-view>_seed<seed>[_<qualifier>]
```

Examples:

```text
b0_runtime_smoke_autodl
a0_adapter_dev_seed11171
a3_combined_paper_v1_seed11172
unet2d_paper_v1_seed11171
```

Never reuse a run ID after changing code, data, configuration, or checkpoint.
Recovery of the same interrupted job may continue under the same run ID when
the resume source is recorded.

## Required Run Manifest

Each run writes a machine-readable manifest before training or evaluation
starts. At minimum it records:

- run ID, start time, stage, method, and declared purpose;
- code revision plus dirty-worktree state;
- split-manifest path and SHA-256;
- train/validation/test case counts and data-view identity;
- base, resume, detector, and final checkpoint SHA-256 values when applicable;
- full resolved CLI/configuration;
- training seed, dataset seed, and deterministic-policy flags;
- GPU, Python, CUDA, PyTorch, package versions, and Conda environment;
- preprocessing, label mapping, prompt condition, and postprocess condition;
- batch size, workers, cache policy, AMP, and cuDNN settings;
- output paths for logs, checkpoints, predictions, per-case metrics, and
  aggregate summaries.

At completion, append wall-clock time, peak allocated/reserved GPU memory,
exit status, selected checkpoint identity, and artifact hashes.

## Artifact Layout

Use a stable per-run layout:

```text
<run-root>/<run-id>/
  manifest.json
  config/
  logs/
  checkpoints/
  predictions/
  metrics/
    per_case.json
    per_case.csv
    summary.json
    summary.md
  plots/
```

Smoke-test artifacts use the same shape so promotion checks do not depend on a
separate code path. A smoke Dice value is a runtime-health signal and must not
be copied into a paper results table.

## Runtime Qualification Gate

The exact long-run code path and resolved configuration must pass these stages
on AutoDL, in order:

1. Imports, dataset integrity, model load, and checkpoint compatibility.
2. One forward pass, backward pass, and optimizer update.
3. One slice-limited epoch that writes the required artifact layout.
4. A fixed 100-update profile on the development training view.
5. A three-epoch stability run using the selected runtime configuration.

Failure at any stage blocks the corresponding long run. A prior group-server
smoke or an implementation-only unit test does not satisfy the AutoDL gate.

## One-Time AutoDL Profile

Performance selection is infrastructure work, not a research sweep. Run it
once before A0 on the same cached 100-update development slice sequence.

- Profile batch sizes 8, 12, and 16 with worker counts 2, 4, and 8.
- Begin with AMP disabled and `cudnn_benchmark=false`.
- Record actual samples/s, batch wait, host-to-device time, GPU compute time,
  peak allocated/reserved memory, and sampled GPU utilization.
- Select the fastest stable configuration below 22 GiB peak allocated memory.
- Compare AMP only through one matched stability and metric-equivalence gate.
- Freeze batch, workers, cache LRU, pin memory, persistent workers, AMP, and
  cuDNN policy for A0-A3 after selection.

`CUDA_LAUNCH_BLOCKING=1` is allowed only for diagnosis and is forbidden in
throughput measurements. Do not run concurrent GPU work while profiling.

## Shared Cache Contract

The versioned cache contains per-case four-modality tensors normalized with
the same per-volume nonzero-minmax transform used by inference, plus compact
segmentation labels.

- Cache metadata records schema version, source case identity, modalities,
  normalization, dtype, shape, source fingerprint, and creation code revision.
- Training augmentation remains online after cache loading.
- Cache validation compares cached tensors against direct inference
  preprocessing within the declared float16 tolerance.
- Incompatible schema, dtype, shape, normalization, or source identity fails
  closed; it must never trigger an implicit rebuild inside a paper run.
- Rebuild the cache from `paper_v1` on AutoDL. Retired-server timing results do
  not select current batch or worker settings.

Negative-slice sampling is method behavior rather than a cache property. Its
ratio, seed, and prompt behavior belong to the training-method child PRD.

## Seed Contract

| Purpose | Seed |
| --- | ---: |
| Paper split, detector data preparation, and first development run | 11171 |
| Primary method seed 2 | 11172 |
| Optional third seed | 11173 |

Seed Python, NumPy, Torch, DataLoader generators/workers, dataset sampling,
prompt jitter, and augmentation. Record deterministic Torch/cuDNN settings.
The same method seed denotes the same stochastic sources across A0-A3.

## Checkpoint Selection

- Default SAM Adapter storage keeps only the retained `best_model.pth`; do not
  pass `--save_epochs` in normal A0/A1/A2/A3 launches.
- Extra epoch snapshots are allowed only for a predeclared short diagnostic and
  should be deleted or moved out of the active run after their metrics are
  captured.
- Slice-level validation metrics are health diagnostics only.
- Select SAM and conventional U-Net checkpoints with the parent PRD's raw
  patient-level macro-Dice endpoint and ET guardrail on validation.
- All methods in one comparison use the same checkpoint-selection prompt
  condition. Do not select A0 with oracle prompts and A3 with YOLO prompts.
- A checkpoint file named `best` is not sufficient provenance; record its
  source epoch, selection metric, per-class guardrails, and SHA-256.

Method-specific epoch budgets and early-stopping rules belong to child PRDs.

## Frozen Automatic Prompts

The detector child task produces one versioned prediction set keyed by case ID
and slice ID. A0-A3 and all final seeds replay that exact set.

- Do not invoke YOLO online during SAM comparisons.
- Do not retrain or reselect YOLO per SAM method or seed.
- Validate prediction coverage against the requested case manifest before
  inference.
- Generate locked-test detector predictions only after the final freeze record
  exists, then reuse them for every final method.

## Configuration Freeze and Test Unlock

Before any test label or test-derived metric is read, create a versioned freeze
record containing:

1. Selected Plan A or Plan B and the satisfied decision gates.
2. Exact code revision and environment identity.
3. Data and development-subset manifest hashes.
4. All model, detector, and checkpoint identities.
5. Training, inference, prompt, threshold, and postprocess configurations.
6. Required methods, seeds, metrics, tests, strata, and figure definitions.
7. Predeclared failure-case selection rules.
8. Date and explicit statement that no test-derived choice has been made.

After unlock, a code or configuration change that could affect predictions or
analysis invalidates the freeze. Do not rerun test selectively; document the
failure and decide whether the entire locked evaluation must be repeated.

## Compute Order and Cut Policy

Run one long GPU job at a time. The default critical-path order is:

1. Runtime/profile gate and A0.
2. Prompt-gap diagnosis and frozen detector predictions.
3. A1/A2/A3 development ablation.
4. Conventional 2D U-Net.
5. Plan A or Plan B freeze and required final seeds.
6. Optional nnU-Net and third seed only when P0 evidence is secure.

When schedule pressure occurs, cut P2 first and then P1. Do not reduce the
frozen train split, reuse validation results as test evidence, or remove an A1
or A2 development arm to save time.
