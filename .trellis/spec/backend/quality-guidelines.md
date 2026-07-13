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
conda run -n "sam-med2d-verify" python -B -m unittest discover -s "tests" -v
```

**Validation and Error Matrix**:

| Condition | Required action |
| --- | --- |
| `conda run` cannot find `sam-med2d-verify` | Stop and report the missing environment; do not substitute system Python. |
| Imports for `torch`, `scipy`, or `nibabel` fail | Report the environment as incomplete; do not classify as a source regression. |
| Unit test fails after imports succeed | Treat it as a code regression and investigate. |

**Cases**:

* Good: `conda run -n "sam-med2d-verify"` executes the relevant test command.
* Base: syntax-only checks may use `python -B` when imports are not needed.
* Bad: running `python -m unittest` from an arbitrary system interpreter.

**Required Tests**: Test reports must state the exact Conda environment and command used.

#### Wrong

```powershell
python -m unittest discover -s "tests"
```

#### Correct

```powershell
conda run -n "sam-med2d-verify" python -B -m unittest discover -s "tests" -v
```

---

## Code Review Checklist

<!-- What reviewers should check -->

(To be filled by the team)
