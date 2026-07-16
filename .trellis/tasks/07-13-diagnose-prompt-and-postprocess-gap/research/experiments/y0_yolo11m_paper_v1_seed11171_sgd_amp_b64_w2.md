# Experiment Report: `y0_yolo11m_paper_v1_seed11171_sgd_amp_b64_w2`

## Verdict

| Field | Value |
| --- | --- |
| Status | `succeeded` |
| Decision | `reject` |
| Profile | `development_screen` |
| Owning task | `07-13-diagnose-prompt-and-postprocess-gap` |
| Root research task | `07-13-icassp-2027-hierarchical-prompt-segmentation` |

**Finding:** YOLO11m training completed successfully, but the best Y3 operating point (`best.pt`, `conf=0.001`) failed the frozen coverage and consecutive-miss gates while producing a `0.5336` background false-positive rate, so this detector must not be frozen or advanced to A0.

## Reproducibility

| Field | Value | Evidence |
| --- | --- | --- |
| Purpose | Y2 formal YOLO11m training followed by Y3 detector selection | `outputs/autodl-runs/y0_yolo11m_paper_v1_seed11171_sgd_amp_b64_w2/manifest.json`; Y3 summaries below |
| Code revision / dirty state | Remote revision and dirty state `missing`; training entrypoint SHA-256 `04602b73...e4bf`; evaluator SHA-256 `95baee02...b431` matched local and AutoDL | training manifest; `src/sam_med2d_finetune/tools/evaluate_yolo_recall.py` |
| Data view / cases | train: 875 cases / 76,180 slices; val: 187 cases / 28,985 slices (11,950 positive, 17,035 negative); test excluded | training manifest; Y3 summaries |
| Split manifest / SHA-256 | `08eb0ac8f25697297d92b0bcb40510bdead5700ed63cf95357e649ecf4850b6f` | training manifest; both Y3 summaries |
| Method / prompt / postprocess | YOLO11m, one Tumor class; Y3 detector-only top-1 box, `max_det=1`, no SAM or postprocess | training manifest; Y3 summaries |
| Seeds | dataset/training seed `11171`, deterministic training enabled | training manifest |
| Command / resolved config | train: `imgsz=320`, `batch=64`, SGD, AMP, workers 2, epochs 100, patience 20; Y3: `conf={0.001,0.003,0.005,0.01,0.03,0.05,0.10}`, NMS IoU `0.60`, `imgsz=320`, batch 64 | `args.yaml`; Y3 summaries |
| Checkpoints / hashes | base `yolo11m.pt`: `d5ffc1a6...5b95`; best: `dbbc426f...fa3`; last: `a4a628f0...647d` | training manifest; Y3 summaries |
| Runtime / environment | train `9,806.13 s` (2.72 h), Y3 `882.1 s` total; RTX 4090 D; Python 3.11.15, Torch 2.6.0+cu124, Ultralytics 8.4.95; peak allocated/reserved `7.91/11.14 GB` | training manifest; command completion records |

## Primary Results

| Metric | Candidate | Baseline | Delta | Evidence |
| --- | ---: | ---: | ---: | --- |
| Raw patient-level macro-Dice | `missing` | `missing` | `missing` | A0 was not run because Stage A failed |
| Raw ET Dice | `missing` | `missing` | `missing` | A0 was not run because Stage A failed |
| Raw TC Dice | `missing` | `missing` | `missing` | A0 was not run because Stage A failed |
| Raw WT Dice | `missing` | `missing` | `missing` | A0 was not run because Stage A failed |
| Raw HD95 (mm) | `missing` | `missing` | `missing` | A0 was not run because Stage A failed |

Paired uncertainty/statistics: `missing` (not applicable before a detector passes Stage A).

## Guardrails

| Guardrail | Result | Interpretation | Evidence |
| --- | --- | --- | --- |
| Y2 detector health at selected epoch 27 | precision `0.93217`, recall `0.82954`, mAP50 `0.89145`, mAP50-95 `0.71148`; early stopped after epoch 47 | Training converged and produced valid checkpoints; these metrics do not select the prompt operating point | `results.csv`; training manifest |
| Fully missed validation cases | `0 / 187` at `best.pt`, `conf=0.001` | Passes the zero-case-miss gate | `best_val/scan_summary.json` |
| Positive-slice coverage recall at 0.50 | `0.91473` (`10,931 / 11,950`), with 1,019 missed slices | Fails the required `>=0.98` gate by 6.53 percentage points | `best_val/scan_summary.json` |
| Maximum consecutive coverage misses | `24` slices | Fails the required `<=2` gate | `best_val/scan_summary.json` |
| Background false-positive slice rate | `0.53361` (`9,090 / 17,035`) | The recall-maximizing threshold heavily contaminates background slices | `best_val/scan_summary.json` |
| Any-box recall / coverage@0.80 / area ratio | `0.99757 / 0.81941 / 2.83962` | Near-complete box emission did not produce adequate target coverage; the dominant failure is localization, not confidence thresholding | `best_val/scan_summary.json` |
| Case distribution at best operating point | 127/187 cases below 0.98 coverage; 78/187 cases exceed two consecutive misses | Failure is broad, not isolated to one outlier | `best_val/scan_summary.json` |
| `last.pt` best operating point | coverage `0.91046`, 1,070 missed slices, maximum run `38`, BG FP `0.51130` | Worse than `best.pt` under the frozen lexicographic ranking | `last_val/scan_summary.json` |

Secondary raw/post analysis: all 14 prediction exports exist on AutoDL and their actual SHA-256 values match the hashes recorded in the two scan summaries. The global top two Stage A points are `best.pt/conf=0.001` and `best.pt/conf=0.003`; neither passes the gate, so Stage B was not started.

## Failures And Missing Evidence

- Failure stage: Y3 Stage A acceptance gate (execution itself succeeded).
- Error or interruption: none.
- Last valid evidence: complete 14-point Y3 grid over all 187 validation cases.
- Missing/conflicting evidence: raw A0 patient-level Dice, class Dice, HD95 and paired statistics are missing because no Stage A candidate qualified.
- Conclusion limit: the detector checkpoint/confidence cannot be frozen, and current automatic-prompt results cannot be promoted to paper-facing evaluation.

## Decision And Next Action

- Decision rationale: reject the current detector candidate because it fails two predeclared hard gates; zero fully missed cases alone is insufficient, and lowering confidence already causes excessive background FP without resolving coverage.
- Next action: run the one permitted recall-oriented YOLO11m retraining at `imgsz=512`, hardware batch `32` with effective batch `64`, retaining SGD, AMP, workers 2, epochs 100 and patience 20, then repeat the unchanged Y3 scan.

## Artifacts

- Manifest: `/root/autodl-tmp/runs/y0_yolo11m_paper_v1_seed11171_sgd_amp_b64_w2/manifest.json`; local mirror `outputs/autodl-runs/y0_yolo11m_paper_v1_seed11171_sgd_amp_b64_w2/manifest.json`
- Metrics: `outputs/autodl-runs/y0_yolo11m_paper_v1_seed11171_sgd_amp_b64_w2/results.csv`; `outputs/autodl-runs/y0_yolo11m_paper_v1_seed11171_sgd_amp_b64_w2/y3_detector_selection/{best_val,last_val}/scan_summary.{json,csv,md}`
- Logs: `outputs/autodl-runs/y0_yolo11m_paper_v1_seed11171_sgd_amp_b64_w2/stdout.log` (local copy is incomplete); terminal status is authoritative in the manifest
- Checkpoints: `/root/autodl-tmp/runs/y0_yolo11m_paper_v1_seed11171_sgd_amp_b64_w2/weights/{best.pt,last.pt}`
- Predictions/plots: `/root/autodl-tmp/runs/y0_yolo11m_paper_v1_seed11171_sgd_amp_b64_w2/y3_detector_selection/{best_val,last_val}/predictions/`; 14 exports, 107 MB total

## Change History

| Date | Transition | Evidence |
| --- | --- | --- |
| `2026-07-16` | `initial -> succeeded / reject` | training manifest plus complete Y3 best/last threshold scans |
