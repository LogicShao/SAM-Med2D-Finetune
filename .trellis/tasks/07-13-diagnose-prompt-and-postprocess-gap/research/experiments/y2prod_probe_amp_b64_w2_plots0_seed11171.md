# Experiment Report: `y2prod_probe_amp_b64_w2_plots0_seed11171`

## Verdict

| Field | Value |
| --- | --- |
| Status | `interrupted` |
| Decision | `retain` |
| Profile | `diagnostic` |
| Owning task | `07-13-diagnose-prompt-and-postprocess-gap` |
| Root research task | `07-13-icassp-2027-hierarchical-prompt-segmentation` |

**Finding:** Disabling plots did not improve the 100-epoch auto/MuSGD production workload, which remained about `3.4 it/s` with two workers.

## Reproducibility

| Field | Value | Evidence |
| --- | --- | --- |
| Purpose | `Y2_production_load_runtime_probe` | `outputs/autodl-runs/y2prod_probe_amp_b64_w2_plots0_seed11171/manifest.json` |
| Code / data / seed | entrypoint SHA `8cf80f72...`; frozen 875/187 split; seed `11171` | same manifest |
| Config | YOLO11m, epochs 100, batch 64, workers 2, AMP true, cache false, plots false, optimizer auto | same manifest |
| Runtime | intentional 107.737 s probe; peak allocated 7,903,088,128 B | same manifest |

## Primary Results

All patient-level segmentation metrics and paired statistics: `missing` and not applicable.

## Guardrails

| Guardrail | Result | Interpretation | Evidence |
| --- | --- | --- | --- |
| Production-load throughput | about `3.4 it/s`; 303/1191 batches at about 91 s training time | Plotting was not the bottleneck. | `outputs/autodl-runs/y2prod_probe_amp_b64_w2_plots0_seed11171/stdout.log` |

## Failures And Missing Evidence

- Error or interruption: intentional SIGTERM after the fixed-duration probe; exit 143.
- Last valid evidence: finite losses through batch 303.
- Conclusion limit: no detector-quality claim.

## Decision And Next Action

- Decision rationale: Retain workers 2 as the worker baseline and investigate optimizer selection.
- Next action: Compare the same workload with workers 8 while keeping all other settings fixed.

## Artifacts

- Manifest: `outputs/autodl-runs/y2prod_probe_amp_b64_w2_plots0_seed11171/manifest.json`
- Logs: `outputs/autodl-runs/y2prod_probe_amp_b64_w2_plots0_seed11171/stdout.log`
- Metrics/checkpoints: `missing` by probe design.

## Change History

| Date | Transition | Evidence |
| --- | --- | --- |
| `2026-07-15` | `initial -> interrupted` | manifest |
