# Repository Guidelines

## Project Structure & Module Organization
- Top-level entry points are `train_singletask.py`, `train_multitask.py`, `evaluate_baseline.py`, `preprocess_brats.py`, and `create_subset.py`.
- `segment_anything/` contains the in-repo SAM-Med2D model code. Keep interface changes localized here.
- `finetune_scripts/` stores reproducible launch wrappers split into `single_task/` and `multi_task/`.
- `pretrain_model/` holds checkpoints. `data_demo/` contains sample processed data. `data_brats_*` directories are raw or processed datasets. `workdir_*` directories contain generated logs, models, and plots and should not be committed.

## Build, Test, and Development Commands
- Install `torch` and `torchvision` for your local CUDA or CPU environment first, then run `pip install -r requirements.txt` for the remaining dependencies.
- `python create_subset.py --source_root data_brats_raw_all --dest_root data_brats_raw --train_num 20 --val_num 4` creates a small BraTS subset for smoke tests.
- `python preprocess_brats.py --train_data_path data_brats_raw/train --val_data_path data_brats_raw/val --processed_data_path data_brats_WT_TC --labels WT TC` converts BraTS NIfTI cases into PNG slices and JSON mappings.
- `python train_singletask.py --finetune_method adapter --data_path data_brats_WT_TC --work_dir workdir_label_WT_TC` runs single-task finetuning.
- `python train_multitask.py --finetune_method lora --train_data_path data_brats_raw/train --val_data_path data_brats_raw/val --work_dir workdir_multi_task` runs multi-task finetuning.
- `python evaluate_baseline.py --data_path data_brats_WT_TC --work_dir workdir_label_WT_TC` evaluates the base checkpoint.
- Use `python finetune_scripts/single_task/adapter.py` or `python finetune_scripts/multi_task/lora.py` for the repo's canned experiment settings.

## Coding Style & Naming Conventions
- Use 4-space indentation, `snake_case` for functions, variables, and files, and `PascalCase` for classes.
- Follow the existing import grouping and explicit CLI argument style built on `argparse`.
- Prefer small helper functions over repeating train and validation logic across scripts.
- The repo does not define `black`, `ruff`, or `pytest` config yet. Keep changes black-compatible, imports tidy, and remove dead code before review.

## Testing Guidelines
- There is no dedicated `tests/` package yet. Validate changes with small-subset smoke tests and metric checks.
- Before opening a PR, run `python -m compileall .` and at least one relevant training or evaluation command.
- For training changes, confirm that `workdir_*/logs`, `workdir_*/models`, and `workdir_*/plots` are produced as expected.
- If you add automated tests, place them under `tests/` and name them `test_<module>.py`.

## Commit & Pull Request Guidelines
- Recent history uses short imperative subjects such as `Add dataset splitting...` and `Refactor dataset preprocessing...`. Keep the same style and keep each commit focused on one concern.
- PRs should state dataset path assumptions, key CLI arguments, affected finetuning method, and any Dice or IoU change. Include plots or screenshots when training curves change.
- Call out any checkpoint, path, or data-format migration explicitly so existing experiment scripts do not break silently.
