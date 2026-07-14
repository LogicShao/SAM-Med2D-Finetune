# Directory Structure

> How backend code is organized in this project.

---

## Overview

The active codebase uses a `src/` layout. New Python modules belong under
`src/sam_med2d_finetune/` unless they are part of the vendored SAM-Med2D
implementation, which lives under `src/segment_anything/`.

Do not add new root-level Python entry points. Run source code with
`PYTHONPATH=src python -m ...` or through the maintained wrappers in
`finetune_scripts/multi_task/`.

---

## Directory Layout

```
src/
├── sam_med2d_finetune/
│   ├── brats/          # BraTS case IO, cache contract, constants, metrics
│   ├── inference/      # Whole-case inference, postprocess, HTML preview
│   ├── models/         # Model construction and checkpoint loading
│   ├── tools/          # Dataset split, cache build, YOLO helper utilities
│   ├── training/       # Active multitask training entry point and dataset
│   ├── utils/          # Shared CLI/training helpers
│   └── web_demo/       # FastAPI demo and background pipeline
└── segment_anything/   # Vendored SAM-Med2D implementation

finetune_scripts/
└── multi_task/         # Stable wrappers for the active training workflow

tests/                  # Unit tests for active modules
.trellis/spec/          # Shared project conventions for future sessions
```

---

## Module Organization

- Put business logic in the package that owns the workflow:
  `training/`, `inference/`, `web_demo/`, `tools/`, or `brats/`.
- Keep CLI parsing in the entry module and move reusable logic to helpers in
  the same package before creating a new cross-package utility.
- `web_demo/services/` owns orchestration and file-system side effects.
  `web_demo/ui/` owns routing and template wiring only.
- Keep `segment_anything/` changes minimal and localized. Adapt project code in
  `sam_med2d_finetune.models` before changing vendored internals.

---

## Naming Conventions

- Use `snake_case` for modules and functions, `PascalCase` for classes.
- Keep package names task-oriented and concrete: `inference/postprocess.py`,
  `training/train_multitask.py`, `tools/split_brats_dataset.py`.
- Prefer explicit module names over generic names at the package boundary. For
  example, use `models/factory.py` instead of `models/utils.py` when the file
  owns model construction.
- Generated assets do not belong in `src/`. Keep outputs under `outputs/`,
  temporary uploads under `outputs/web_demo_runs/`, and large binary data out of
  the repository.

---

## Examples

- `src/sam_med2d_finetune/training/train_multitask.py` is the canonical active
  training entry point.
- `src/sam_med2d_finetune/inference/volume.py` and
  `src/sam_med2d_finetune/inference/batch_validate.py` show the expected split
  between single-case inference and evaluation.
- `src/sam_med2d_finetune/web_demo/services/pipeline.py` is the reference for
  subprocess orchestration that must re-export `PYTHONPATH=src`.
