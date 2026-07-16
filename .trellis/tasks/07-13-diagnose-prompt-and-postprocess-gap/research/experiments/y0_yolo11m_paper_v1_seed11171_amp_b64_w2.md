# Experiment Report: `y0_yolo11m_paper_v1_seed11171_amp_b64_w2`

## Verdict

| Field | Value |
| --- | --- |
| Status | `interrupted` |
| Decision | `reject` |
| Profile | `failure` |
| Owning task | `07-13-diagnose-prompt-and-postprocess-gap` |
| Root research task | `07-13-icassp-2027-hierarchical-prompt-segmentation` |

**Finding:** The 100-epoch auto-optimizer run was intentionally stopped after three complete epochs because Ultralytics 8.4.95 selected MuSGD and required about 400 s per epoch, roughly twice the validated SGD throughput.

## Reproducibility

| Field | Value | Evidence |
| --- | --- | --- |
| Purpose | `Y2_formal_YOLO11m_optimized_runtime_gate_20260715` | `outputs/autodl-runs/y0_yolo11m_paper_v1_seed11171_amp_b64_w2/manifest.json` |
| Code revision / dirty state | revision and dirty state `null`; entrypoint SHA-256 `8cf80f72...` | same manifest |
| Data view / cases | frozen YOLO `paper_v1`; train 875, val 187 | same manifest |
| Split manifest / SHA-256 | `08eb0ac8f25697297d92b0bcb40510bdead5700ed63cf95357e649ecf4850b6f` | same manifest |
| Method / prompt / postprocess | YOLO11m WT detector; no SAM evaluation | same manifest |
| Seeds | `11171` | same manifest |
| Command / resolved config | epochs 100, batch 64, workers 2, AMP true, cache false, optimizer auto resolved to MuSGD | args; stdout |
| Checkpoints / hashes | base `d5ffc1a6...`; interrupted best `787b5fe9...`; last `415cdd37...` | manifest |
| Runtime / environment | 1230.612 s wall; exit 143; peak allocated 8,025,387,008 B; reserved 11,169,431,552 B | manifest |

## Primary Results

Raw patient-level Dice, ET/TC/WT Dice, HD95, and paired statistics: `missing` (training did not reach detector selection or SAM evaluation).

## Guardrails

| Guardrail | Result | Interpretation | Evidence |
| --- | --- | --- | --- |
| Completed epochs | 3 | Interruption occurred after valid epoch rows were written. | `outputs/autodl-runs/y0_yolo11m_paper_v1_seed11171_amp_b64_w2/results.csv` |
| Runtime | 404.454, 397.572, and 403.234 s per epoch | The projected 100-epoch runtime was about 11.1 h. | results CSV |
| Detector health | epoch-1 mAP50 `0.83904`; epoch-3 `0.80840` | Secondary health metrics only; no checkpoint selection claim. | results CSV |

## Failures And Missing Evidence

- Failure stage: Y2 detector training runtime qualification.
- Error or interruption: deliberate SIGTERM, recorded as `TrainingTermination`, exit 143.
- Last valid evidence: three complete train/val rows and best/last checkpoints.
- Missing/conflicting evidence: no completed Y2 trajectory or Y3 selection scan.
- Conclusion limit: the run cannot select or promote a detector checkpoint.

## Decision And Next Action

- Decision rationale: Reject the MuSGD runtime configuration, not YOLO11m, because the newer Ultralytics auto policy doubled wall time without being part of the frozen detector hypothesis.
- Next action: Restart from the verified base checkpoint with explicit standard SGD settings.

## Artifacts

- Manifest: `outputs/autodl-runs/y0_yolo11m_paper_v1_seed11171_amp_b64_w2/manifest.json`
- Metrics: `outputs/autodl-runs/y0_yolo11m_paper_v1_seed11171_amp_b64_w2/results.csv`
- Logs: `outputs/autodl-runs/y0_yolo11m_paper_v1_seed11171_amp_b64_w2/stdout.log`
- Checkpoints: remote paths and hashes in manifest; not pulled locally.

## Change History

| Date | Transition | Evidence |
| --- | --- | --- |
| `2026-07-15` | `initial -> interrupted` | manifest |
