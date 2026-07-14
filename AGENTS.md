# Repository Guidelines

## Project Structure & Module Organization
- Active Python code lives under `src/sam_med2d_finetune/` and is invoked with `python -m ...` after setting `PYTHONPATH=src`.
- `src/segment_anything/` contains the in-repo SAM-Med2D model code. Keep interface changes localized there.
- `finetune_scripts/multi_task/` stores reproducible launch wrappers for the current training path.
- `pretrain_model/` holds checkpoints. `data_demo/` contains sample processed data. `data_brats_*` directories are raw or processed datasets. `workdir_*` directories contain generated logs, models, and plots and should not be committed.

## Build, Test, and Development Commands
- Install `torch` and `torchvision` for your local CUDA or CPU environment first, then run `pip install -r requirements.txt` for the remaining dependencies.
- `PYTHONPATH=src python -m sam_med2d_finetune.tools.split_brats_dataset --source_dir data_brats_raw_all --out_dir data_brats_raw --dev_out_dir data_brats_dev --dev_size 20 --clean` creates a reproducible BraTS split plus a small dev subset.
- `PYTHONPATH=src python -m sam_med2d_finetune.tools.precache_brats_cases --cases_root data_brats_raw/train --cache_root temp/brats_cache --max_cases 4` builds a small mmap cache for smoke tests.
- `PYTHONPATH=src python -m sam_med2d_finetune.training.train_multitask --finetune_method lora --train_data_path data_brats_raw/train --val_data_path data_brats_raw/val --work_dir workdir_multi_task` runs multi-task finetuning.
- `PYTHONPATH=src python -m sam_med2d_finetune.inference.volume --case_dir data_brats_raw/val/BraTS2021_xxxxx --output_dir outputs/demo_case --sam_checkpoint pretrain_model/sam-med2d_b.pth --finetuned_checkpoint workdir_multi_task/models/finetune_adapter/best_model.pth --finetune_method adapter` runs whole-case inference.
- `PYTHONPATH=src python -m sam_med2d_finetune.inference.batch_validate --cases_root data_brats_raw/val --output_root outputs/validation_run --sam_checkpoint pretrain_model/sam-med2d_b.pth --finetuned_checkpoint workdir_multi_task/models/finetune_adapter/best_model.pth --finetune_method adapter` evaluates the active whole-case pipeline.
- Use `python finetune_scripts/multi_task/adapter.py` or `python finetune_scripts/multi_task/lora.py` for the repo's canned experiment settings.

## Coding Style & Naming Conventions
- Use 4-space indentation, `snake_case` for functions, variables, and files, and `PascalCase` for classes.
- Follow the existing import grouping and explicit CLI argument style built on `argparse`.
- Prefer small helper functions over repeating train and validation logic across scripts.
- The repo does not define `black`, `ruff`, or `pytest` config yet. Keep changes black-compatible, imports tidy, and remove dead code before review.

## Testing Guidelines
- There is no dedicated `tests/` package yet. Validate changes with small-subset smoke tests and metric checks.
- Before opening a PR, set `PYTHONPATH=src`, run `python -m compileall src tests finetune_scripts`, and execute at least one relevant training or evaluation command.
- For training changes, confirm that `workdir_*/logs`, `workdir_*/models`, and `workdir_*/plots` are produced as expected.
- If you add automated tests, place them under `tests/` and name them `test_<module>.py`.

## Commit & Pull Request Guidelines
- Recent history uses short imperative subjects such as `Add dataset splitting...` and `Refactor dataset preprocessing...`. Keep the same style and keep each commit focused on one concern.
- PRs should state dataset path assumptions, key CLI arguments, affected finetuning method, and any Dice or IoU change. Include plots or screenshots when training curves change.
- Call out any checkpoint, path, or data-format migration explicitly so existing experiment scripts do not break silently.
