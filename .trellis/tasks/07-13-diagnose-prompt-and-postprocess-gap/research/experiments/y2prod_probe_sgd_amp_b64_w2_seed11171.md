# Experiment Report: `y2prod_probe_sgd_amp_b64_w2_seed11171`

## Verdict

| Field | Value |
| --- | --- |
| Status | `interrupted` |
| Decision | `promote` |
| Profile | `diagnostic` |
| Owning task | `07-13-diagnose-prompt-and-postprocess-gap` |
| Root research task | `07-13-icassp-2027-hierarchical-prompt-segmentation` |

**Finding:** Explicit standard SGD restored the same 100-epoch production workload to about `6.9 it/s`, roughly twice the MuSGD throughput, with finite AMP losses and slightly lower peak allocation.

## Reproducibility

| Field | Value | Evidence |
| --- | --- | --- |
| Purpose | `Y2_production_load_SGD_runtime_probe` | `outputs/autodl-runs/y2prod_probe_sgd_amp_b64_w2_seed11171/manifest.json` |
| Code / data / seed | entrypoint SHA `04602b73...`; frozen 875/187 split; seed `11171` | same manifest |
| Config | YOLO11m, epochs 100, batch 64, workers 2, AMP true, cache false, optimizer SGD, lr0 0.01, momentum 0.9, warmup bias LR 0.0 | same manifest |
| Runtime | intentional 108.166 s probe; peak allocated 7,825,129,984 B; reserved 8,296,333,312 B | same manifest |

## Primary Results

All patient-level segmentation metrics and paired statistics: `missing` and not applicable.

## Guardrails

| Guardrail | Result | Interpretation | Evidence |
| --- | --- | --- | --- |
| Production-load throughput | about `6.9 it/s`; 626/1191 batches at about 93 s training time | About 2.0x the MuSGD workers-2 probe. | `outputs/autodl-runs/y2prod_probe_sgd_amp_b64_w2_seed11171/stdout.log` |
| AMP stability | finite losses through batch 626 | No NaN or OOM observed. | stdout |

## Failures And Missing Evidence

- Error or interruption: intentional SIGTERM after the fixed-duration probe; exit 143.
- Last valid evidence: finite losses through batch 626.
- Conclusion limit: runtime promotion only; no detector-quality claim.

## Decision And Next Action

- Decision rationale: Promote this single runtime configuration because it restores traditional YOLO11 long-training SGD semantics and materially improves throughput without changing model, data, effective batch, augmentation, or image size.
- Next action: Start a fresh 100-epoch Y2 run from the verified YOLO11m base checkpoint.

## Artifacts

- Manifest: `outputs/autodl-runs/y2prod_probe_sgd_amp_b64_w2_seed11171/manifest.json`
- Logs: `outputs/autodl-runs/y2prod_probe_sgd_amp_b64_w2_seed11171/stdout.log`
- Metrics/checkpoints: `missing` by probe design.

## Change History

| Date | Transition | Evidence |
| --- | --- | --- |
| `2026-07-15` | `initial -> interrupted` | manifest |
