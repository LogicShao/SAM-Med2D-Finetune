# Reorganize Source Layout and Archive Legacy Code

## Goal

Move the active SAM-Med2D finetuning code into a clear `src/` package layout,
remove the root-level script layout, and keep one active workflow that matches
the current paper-oriented Trellis plan.

## Scope

- Create `src/sam_med2d_finetune/` as the only active project package.
- Move the vendored SAM-Med2D implementation into `src/segment_anything/`.
- Keep active BraTS, training, inference, model, utility, tool, and web demo
  modules inside `src/`.
- Update scripts, tests, and the web demo to call package modules through
  `python -m ...` with `PYTHONPATH=src`.
- Remove old PNG/single-task pipeline scripts instead of preserving stale root
  compatibility wrappers.
- Update Trellis and repository documentation so later sessions immediately see
  the new entry points and directory boundaries.

## Active Code Boundary

Active paper-track modules include:

- Multi-task training and dataset code.
- BraTS case/cache/constants/metrics code.
- 3D inference, batch validation, prompt strategy, postprocessing, and
  visualization code.
- Model factory and CLI utilities used by the active path.
- Web demo pipeline, results view, and background job orchestration.

Legacy candidates include:

- `DataLoader.py`
- `metrics.py`
- `utils.py`
- `train_singletask.py`
- `evaluate_baseline.py`
- `preprocess_brats.py`
- `create_subset.py`
- `split_raw_data.py`
- `finetune_scripts/single_task/*`

`tools/prepare_yolo_data.py` is incompatible with the current PRD because it
discovers `test` automatically and applies one background ratio across splits.
Do not silently treat it as paper-compliant in this refactor.

## Compatibility Requirements

- All active commands must resolve through package modules, for example
  `python -m sam_med2d_finetune.training.train_multitask` and
  `python -m sam_med2d_finetune.inference.volume`.
- Tests should import from `sam_med2d_finetune...` rather than deleted root
  scripts.
- Web demo subprocess calls must launch package modules and inject
  `PYTHONPATH=src`.
- Do not move datasets, checkpoints, run outputs, or work directories.
- Do not revert unrelated dirty files.

## Acceptance Criteria

- Active modules live under `src/sam_med2d_finetune/` with explicit package
  imports.
- `src/segment_anything/` is the only vendored SAM package location.
- No active root-level Python wrappers remain.
- Legacy files that conflict with the current plan are deleted from the active
  tree.
- Unit tests pass in the required verification environment with
  `PYTHONPATH=src`.
- Syntax check passes for the new package and maintained wrappers.
