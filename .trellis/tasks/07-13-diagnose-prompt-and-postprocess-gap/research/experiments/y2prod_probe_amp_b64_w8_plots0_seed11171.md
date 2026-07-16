# Experiment Report: `y2prod_probe_amp_b64_w8_plots0_seed11171`

## Verdict

| Field | Value |
| --- | --- |
| Status | `interrupted` |
| Decision | `reject` |
| Profile | `diagnostic` |
| Owning task | `07-13-diagnose-prompt-and-postprocess-gap` |
| Root research task | `07-13-icassp-2027-hierarchical-prompt-segmentation` |

**Finding:** Increasing workers from 2 to 8 reduced the fixed-duration production-load throughput from about `3.4` to `3.1 it/s` without reducing GPU memory.

## Reproducibility

| Field | Value | Evidence |
| --- | --- | --- |
| Purpose | `Y2_production_load_runtime_probe` | `outputs/autodl-runs/y2prod_probe_amp_b64_w8_plots0_seed11171/manifest.json` |
| Code / data / seed | entrypoint SHA `8cf80f72...`; frozen 875/187 split; seed `11171` | same manifest |
| Config | YOLO11m, epochs 100, batch 64, workers 8, AMP true, cache false, plots false, optimizer auto | same manifest |
| Runtime | intentional 107.792 s probe; peak allocated 7,903,087,616 B | same manifest |

## Primary Results

All patient-level segmentation metrics and paired statistics: `missing` and not applicable.

## Guardrails

| Guardrail | Result | Interpretation | Evidence |
| --- | --- | --- | --- |
| Production-load throughput | about `3.1 it/s`; 276/1191 batches at about 90 s training time | Extra workers add overhead on the 15-vCPU quota. | `outputs/autodl-runs/y2prod_probe_amp_b64_w8_plots0_seed11171/stdout.log` |

## Failures And Missing Evidence

- Error or interruption: intentional SIGTERM after the fixed-duration probe; exit 143.
- Last valid evidence: finite losses through batch 276.
- Conclusion limit: no detector-quality claim.

## Decision And Next Action

- Decision rationale: Reject workers 8 because it is slower than the otherwise identical workers-2 probe.
- Next action: Keep workers 2 and isolate the optimizer policy.

## Artifacts

- Manifest: `outputs/autodl-runs/y2prod_probe_amp_b64_w8_plots0_seed11171/manifest.json`
- Logs: `outputs/autodl-runs/y2prod_probe_amp_b64_w8_plots0_seed11171/stdout.log`
- Metrics/checkpoints: `missing` by probe design.

## Change History

| Date | Transition | Evidence |
| --- | --- | --- |
| `2026-07-15` | `initial -> interrupted` | manifest |
