# Experiment Report: `a0_adapter_dev_seed11171`

## Verdict

| Field | Value |
| --- | --- |
| Status | `incomplete_evidence` |
| Decision | `retain` |
| Profile | `development_screen` |
| Owning task | `.trellis/tasks/07-13-implement-hierarchical-prompt-robust-training` |
| Root research task | `.trellis/tasks/07-13-icassp-2027-hierarchical-prompt-segmentation` |

**Finding:** The five-epoch A0 Adapter training trajectory completed and its retained checkpoint is valid as a development proxy, but no patient-level raw 3D result is available for a research decision.

## Reproducibility

| Field | Value | Evidence |
| --- | --- | --- |
| Purpose | Establish the A0 Adapter development baseline before prompt-gap and A1-A3 comparisons. | `../prd.md`; parent checkpoint policy |
| Code revision / dirty state | `missing` | No run manifest was recorded. |
| Data view / cases | Declared 200/43 development view; run-level case manifest `missing` | `../prd.md` |
| Split manifest / SHA-256 | `missing` | No run manifest was recorded. |
| Method / prompt / postprocess | SAM-Med2D Adapter A0; evaluation prompt and postprocess not yet applicable | parent checkpoint policy |
| Seeds | `11171` in the declared run identity; dataset seed evidence `missing` | parent checkpoint policy |
| Command / resolved config | `missing` | No archived resolved configuration was cited. |
| Checkpoints / hashes | retained `best_model.pth`, SHA-256 `97dbb0bca186449c280d15f61e304b6fb797c8e6fff722cb6d31ea4f0b0a7591` | parent checkpoint policy |
| Runtime / environment | peak allocated about 6678 MiB; peak reserved about 7104 MiB; full environment `missing` | parent checkpoint policy |

## Primary Results

| Metric | Candidate | Baseline | Delta | Evidence |
| --- | ---: | ---: | ---: | --- |
| Raw patient-level macro-Dice | `missing` | `missing` | `missing` | Frozen-YOLO 43-case evaluation not run. |
| Raw ET Dice | `missing` | `missing` | `missing` | Frozen-YOLO 43-case evaluation not run. |
| Raw TC Dice | `missing` | `missing` | `missing` | Frozen-YOLO 43-case evaluation not run. |
| Raw WT Dice | `missing` | `missing` | `missing` | Frozen-YOLO 43-case evaluation not run. |
| Raw HD95 (mm) | `missing` | `missing` | `missing` | Frozen-YOLO 43-case evaluation not run. |

Paired uncertainty/statistics: `missing`

## Guardrails

| Guardrail | Result | Interpretation | Evidence |
| --- | --- | --- | --- |
| Epoch 5 slice-level validation Dice | mean `0.8379`; ET `0.8144`; TC `0.8761`; WT `0.8232` | Training remained healthy through the frozen five-epoch budget; this is not a paper endpoint. | parent checkpoint policy |
| Checkpoint retention | only `best_model.pth` retained | The active storage policy was applied after explicit cleanup confirmation. | parent checkpoint policy |

Secondary raw/post analysis: `not applicable`

## Failures And Missing Evidence

- Failure stage: `not applicable`
- Error or interruption: `not applicable`
- Last valid evidence: epoch 5 metrics and retained checkpoint hash.
- Missing/conflicting evidence: run manifest, code state, split hash, resolved config, full environment and 43-case raw 3D evaluation.
- Conclusion limit: slice-level Dice cannot select the paper checkpoint or establish segmentation quality.

## Decision And Next Action

- Decision rationale: retain the checkpoint under the frozen storage policy, but make no candidate-performance claim until raw patient-level evaluation exists.
- Next action: after Y3 freezes YOLO predictions, evaluate this exact checkpoint on the fixed 43 cases with raw output and the unified 3D evaluator.

## Artifacts

- Manifest: `missing`
- Metrics: `metrics.csv` under the run root; exact relative path `missing`
- Logs: `/root/autodl-tmp/runs/a0_adapter_dev_seed11171/`
- Checkpoints: retained `best_model.pth` under the run root; exact relative path `missing`
- Predictions/plots: `missing`

## Change History

| Date | Transition | Evidence |
| --- | --- | --- |
| `2026-07-14` | `initial -> incomplete_evidence` | parent checkpoint policy |
