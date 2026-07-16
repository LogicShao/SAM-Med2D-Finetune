# Quality Guidelines

> Code quality standards for backend development.

---

## Overview

<!--
Document your project's quality standards here.

Questions to answer:
- What patterns are forbidden?
- What linting rules do you enforce?
- What are your testing requirements?
- What code review standards apply?
-->

(To be filled by the team)

---

## Forbidden Patterns

<!-- Patterns that should never be used and why -->

(To be filled by the team)

---

## Required Patterns

<!-- Patterns that must always be used -->

(To be filled by the team)

---

## Testing Requirements

<!-- What level of testing is expected -->

### Convention: Verification Environment

**What**: Run repository syntax checks, unit tests, and training smoke tests through the `sam-med2d-verify` Conda environment.

**Why**: The system Python environment may not contain the CUDA-compatible PyTorch and medical-imaging dependencies required by this project. Verifying there produces import failures unrelated to the code under review.

**Command Contract**:

```powershell
$env:PYTHONPATH="src"
conda run -n "sam-med2d-verify" python -B -m unittest discover -s "tests" -v
```

Source-layout syntax checks should use the same import contract:

```powershell
$env:PYTHONPATH="src"
python -B -m compileall "src" "tests" "finetune_scripts"
```

**Validation and Error Matrix**:

| Condition | Required action |
| --- | --- |
| `conda run` cannot find `sam-med2d-verify` | Stop and report the missing environment; do not substitute system Python. |
| `PYTHONPATH=src` is missing and package imports fail | Treat it as a verification setup error and rerun with the source root exported. |
| Imports for `torch`, `scipy`, or `nibabel` fail | Report the environment as incomplete; do not classify as a source regression. |
| Unit test fails after imports succeed | Treat it as a code regression and investigate. |

**Cases**:

* Good: `conda run -n "sam-med2d-verify"` executes the relevant test command with `PYTHONPATH=src`.
* Base: syntax-only checks may use `python -B` when imports are not needed.
* Bad: running `python -m unittest` from an arbitrary system interpreter.

**Required Tests**: Test reports must state the exact Conda environment and command used.

#### Wrong

```powershell
python -m unittest discover -s "tests"
```

#### Correct

```powershell
$env:PYTHONPATH="src"
conda run -n "sam-med2d-verify" python -B -m unittest discover -s "tests" -v
```

## Scenario: Long-Running Experiment Orchestration CLIs

### 1. Scope / Trigger

- Trigger: adding or changing a backend CLI that launches training, validation,
  artifact synchronization, or another long-running experiment process.
- Applies to tools under `src/sam_med2d_finetune/tools/` that own subprocess
  orchestration or terminal experiment state.

### 2. Signatures

- CLI modules must run through the source layout:

```powershell
$env:PYTHONPATH="src"
python -m sam_med2d_finetune.tools.<module> <subcommand> ...
```

- Long-running controllers should expose a status/readiness command when they
  write asynchronous state, for example:

```powershell
python -m sam_med2d_finetune.tools.run_yolo_retrain_pipeline status --pipeline_dir <dir>
python -m sam_med2d_finetune.tools.run_yolo_retrain_pipeline mark-ready --pipeline_dir <dir> --require_sha <path>=<sha256>
```

### 3. Contracts

- Every launched run must write a machine-readable manifest before starting the
  expensive subprocess.
- Pipeline manifests must include current stage, PID when a subprocess is
  active, update timestamp, selected configuration, terminal result, and
  artifact hashes when available.
- Completion markers are ordered:
  `REMOTE_PIPELINE_COMPLETE` can be written only after remote training and
  evaluation artifacts are complete; `READY_TO_POWER_OFF` can be written only
  after evidence pull/report sync is verified; `PIPELINE_FAILED` is reserved
  for runtime failures such as command failure, OOM, NaN, deadline expiry, or
  missing required artifacts; `SHUTDOWN_REQUESTED` is written only after the
  terminal evidence marker and immediately before invoking the platform
  shutdown command.
- A metric gate rejection with complete evidence is a terminal experiment
  decision, not automatically a runtime failure marker.
- AutoDL automatic shutdown is allowed only through the documented fixed
  `/usr/bin/shutdown` command and only as the final terminal action. Do not
  probe it with `--help`, `--version`, `test`, `ls`, or `sha256sum`; those
  commands are not part of the contract and can trigger shutdown.

### 4. Validation & Error Matrix

| Condition | Required behavior |
| --- | --- |
| Subprocess exits non-zero | Stop downstream stages, update manifest, write `PIPELINE_FAILED` |
| Log contains OOM/NaN terminal signal | Stop downstream stages, update manifest, write `PIPELINE_FAILED` |
| Required checkpoint or Y3 summary is missing | Stop downstream stages, update manifest, write `PIPELINE_FAILED` |
| Formal metric gate fails after complete Y3 | Write `REMOTE_PIPELINE_COMPLETE`, return a distinct non-success code, request shutdown if enabled, do not start another formal run |
| `READY_TO_POWER_OFF` requested before remote completion | Reject the command |
| Required SHA verification fails | Reject the command and do not write readiness marker |
| AutoDL shutdown is enabled and any terminal state is reached | Write `SHUTDOWN_REQUESTED`, flush/sync logs, then execute `/usr/bin/shutdown` with no arguments |
| `/usr/bin/shutdown` execution fails | Write `SHUTDOWN_FAILED` and preserve the original terminal exit code |

### 5. Good/Base/Bad Cases

- Good: a passing screen skips remaining screens, starts exactly one formal run
  from the base checkpoint, scans `best.pt` and `last.pt`, then writes
  `REMOTE_PIPELINE_COMPLETE`.
- Base: a formal run completes but fails its metric gate; the pipeline records
  the terminal rejection, does not write `PIPELINE_FAILED`, and still requests
  shutdown when `--shutdown_on_exit` is enabled.
- Bad: a controller calls `poweroff`, systemd shutdown, kills PID 1, or probes
  `/usr/bin/shutdown --help` to stop billing.

### 6. Tests Required

- Command-construction tests assert fixed hyperparameters and artifact paths.
- Gate tests assert all hard metric criteria, not only one aggregate metric.
- State-machine tests assert early success skips later screens/fallbacks.
- Failure tests assert training failure does not start evaluation.
- Marker tests assert `READY_TO_POWER_OFF` cannot precede
  `REMOTE_PIPELINE_COMPLETE` and requires SHA verification.
- Shutdown tests mock `os.execv` and assert no real shutdown command is executed
  during tests.

### 7. Wrong vs Correct

#### Wrong

```powershell
python train_screen.py
python eval.py
shutdown now
```

#### Correct

```powershell
$env:PYTHONPATH="src"
python -m sam_med2d_finetune.tools.run_yolo_retrain_pipeline run --data <data.yaml> --model <yolo11m.pt>
python -m sam_med2d_finetune.tools.run_yolo_retrain_pipeline status --pipeline_dir <dir>
python -m sam_med2d_finetune.tools.run_yolo_retrain_pipeline mark-ready --pipeline_dir <dir> --require_sha <report>=<sha256>
python -m sam_med2d_finetune.tools.run_yolo_retrain_pipeline run --data <data.yaml> --model <yolo11m.pt> --shutdown_on_exit --shutdown_command /usr/bin/shutdown
```

---

## Code Review Checklist

<!-- What reviewers should check -->

(To be filled by the team)
