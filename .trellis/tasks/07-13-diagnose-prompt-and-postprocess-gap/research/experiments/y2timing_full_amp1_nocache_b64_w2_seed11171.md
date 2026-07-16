# Experiment Report: `y2timing_full_amp1_nocache_b64_w2_seed11171`

## Verdict

| Field | Value |
| --- | --- |
| Status | `succeeded` |
| Decision | `retain` |
| Profile | `diagnostic` |
| Owning task | `07-13-diagnose-prompt-and-postprocess-gap` |
| Root research task | `07-13-icassp-2027-hierarchical-prompt-segmentation` |

**Finding:** The full-data one-epoch AMP gate completed in 301.324 s wall time with finite losses, but its automatically selected AdamW optimizer makes its 222.566 s epoch unsuitable for estimating the 100-epoch production run.

## Reproducibility

| Field | Value | Evidence |
| --- | --- | --- |
| Purpose | `Y2_full_data_one_epoch_timing_gate` | `outputs/autodl-runs/y2timing_full_amp1_nocache_b64_w2_seed11171/manifest.json` |
| Code revision / dirty state | revision `null`; dirty `false` | same manifest |
| Data view / cases | frozen YOLO `paper_v1`; train 875, val 187 | same manifest |
| Split manifest / SHA-256 | `08eb0ac8f25697297d92b0bcb40510bdead5700ed63cf95357e649ecf4850b6f` | same manifest |
| Method / prompt / postprocess | YOLO11m WT detector; no SAM evaluation | same manifest |
| Seeds | `11171` | same manifest |
| Command / resolved config | epochs 1, batch 64, workers 2, AMP true, cache false, optimizer auto resolved to AdamW | `outputs/autodl-runs/y2timing_full_amp1_nocache_b64_w2_seed11171/args.yaml`; stdout |
| Checkpoints / hashes | base `d5ffc1a6...`; best `f592f5b1...`; last `97e58aa9...` | manifest |
| Runtime / environment | 301.324 s wall; peak allocated 7,904,403,968 B; RTX 4090D; Ultralytics 8.4.95 | manifest |

## Primary Results

Raw patient-level Dice, ET/TC/WT Dice, HD95, and paired statistics: `missing` (not applicable to this detector runtime gate).

## Guardrails

| Guardrail | Result | Interpretation | Evidence |
| --- | --- | --- | --- |
| AMP finite-loss gate | passed | No NaN or OOM occurred. | stdout; manifest status |
| Detector health | mAP50 `0.76156`; mAP50-95 `0.49508` | Secondary one-epoch health metric only. | `outputs/autodl-runs/y2timing_full_amp1_nocache_b64_w2_seed11171/results.csv` |
| Runtime comparability | failed | One epoch selected AdamW, while a 100-epoch `optimizer=auto` run selected MuSGD. | args; stdout |

## Failures And Missing Evidence

- Failure stage: not applicable.
- Missing/conflicting evidence: no patient-level segmentation metrics; optimizer differs from production-length auto selection.
- Conclusion limit: this run validates AMP health but cannot predict 100-epoch runtime or select the detector.

## Decision And Next Action

- Decision rationale: Retain as the AMP finite-loss gate only.
- Next action: Freeze one explicit production optimizer before restarting Y2.

## Artifacts

- Manifest: `outputs/autodl-runs/y2timing_full_amp1_nocache_b64_w2_seed11171/manifest.json`
- Metrics: `outputs/autodl-runs/y2timing_full_amp1_nocache_b64_w2_seed11171/results.csv`
- Logs: `outputs/autodl-runs/y2timing_full_amp1_nocache_b64_w2_seed11171/stdout.log`
- Checkpoints: remote run directory in manifest; not pulled locally.

## Change History

| Date | Transition | Evidence |
| --- | --- | --- |
| `2026-07-15` | `initial -> succeeded` | manifest |
