# B0 Cache and B1 Strategy Review

## Outcome

B0 performance work is complete. Cache plus batch 4 is fast enough to begin
research experiments. The next bottleneck is experimental validity rather
than throughput.

## Verified Progress

- A 243-case development cache exists on the server and occupies about 19 GB.
- Cache batch wait fell from 29.4 ms to 6.7 ms at batch 1.
- Batch 4 reached 14.02 samples/s; adding one negative slice per three positive
  slices reached 14.61 samples/s.
- Batch 8 added little throughput, increased data wait and halved optimizer
  updates per epoch, so it is not the default.
- The server reports eight passing tests. Local syntax compilation passes; the
  local Windows Python cannot execute the tests because SciPy and SimpleITK are
  not installed in that interpreter.

## Review Findings

### 1. Patient-level checkpoint selection is not implemented

`train_multitask.py` currently overwrites `best_model.pth` when slice-level
validation mean Dice improves. It does not retain epoch 1/3/5 snapshots. A0
must not start until immutable epoch snapshots can be evaluated retrospectively
with the common raw 3D evaluator.

### 2. Global training determinism is not implemented

`dataset_seed` controls negative-slice sampling and deterministic negative
boxes only. Python, NumPy, Torch, DataLoader shuffling/workers and
Albumentations are not seeded by the training entry point. This blocks a
defensible seed claim even though the data subset itself is deterministic.

### 3. The proposed B1 negative configuration contaminates A0

An empty slice with a random prompt box is a false-positive prompt simulation.
The PRD defines false-positive prompt simulation as PJT, so A0 cannot use
`negative_to_positive_ratio=1/3` plus `negative_prompt_box=random`. Keep that
configuration for A1/A3; use positive slices and oracle class boxes for A0/A2.

### 4. Cache metadata is written but not enforced

The cache builder records schema version, dtype, shape and normalization, but
the dataset loads only `images.npy` and `seg.npy`. Before long training, add a
small compatibility check and a numerical equivalence test against the
inference normalization path. Avoid a broad cache refactor.

### 5. Profiler throughput has a negligible accounting approximation

Samples per second is calculated as `step_count * configured_batch_size`, so
the final partial batch can be overcounted. This is negligible for the current
capacity decision and is not a blocker for A0, but paper-facing runtime tables
should count actual samples.

## Immediate Execution Sequence

1. Add global seed control, immutable epoch snapshots and cache-contract tests.
2. Run the server unit tests and one cached forward/backward smoke.
3. Run one five-epoch A0 trajectory with batch 4, no negative slices, AMP off,
   cuDNN off and GPU1.
4. Evaluate epoch 1/3/5 snapshots on all 43 dev-val cases with upper-bound
   prompts, raw output only and no HTML rendering.
5. Freeze the A0 trajectory or extend only when epoch 5 is clearly still
   improving; then execute the B2 prompt-gap diagnostic before implementing
   additional method components.

## B1 Launch Gate Result (2026-07-13)

The pre-A0 validity fixes are implemented and validated on the GPU1 server.

- `brats_cache.py` is the single source for per-volume nonzero-minmax
  normalization and cache schema constants. Cache construction, training cache
  loading, and whole-volume inference now consume that contract.
- Cache metadata is validated before dataset indexing. Schema version, case ID,
  filenames, dtypes, normalization identifier, dimensions and recorded shapes
  must agree with the memory-mapped arrays.
- `train_multitask.py` has a global `--seed`, seeded Python/NumPy/Torch/CUDA,
  seeded DataLoader generators/workers, and immutable `--save_epochs`
  snapshots. Strict Torch deterministic mode remains opt-in because the SAM
  CUDA cumsum path has no fully deterministic implementation; experiments use
  a fixed seed and controlled sample order, not a bitwise-determinism claim.
- Runtime CSV rows now count the actual number of batch samples.
- Server compilation and all 12 unit tests passed. The test set covers metrics,
  cache metadata rejection, normalization equivalence within float16 tolerance,
  global random-sequence repeatability and immutable snapshots.
- A cached GPU1 backward smoke passed with the final A0 settings: two updates,
  batch 4, `seed=42`, `deterministic=false`, AMP off, cuDNN off, two workers,
  cache LRU 8, no negative slices and zero empty prompts. It created
  `workdir_smoke_contract_gpu1/models/b1_cached_a0_seed42_contract_adapter/epoch_001.pth`.
  Its CSV records 8 actual training samples and 4.16 samples/s. This is a
  pipeline-health result only.

The B1 launch gate is therefore passed. The next action is exactly one
five-epoch A0 trajectory with immutable snapshots at epochs 1, 3 and 5.
