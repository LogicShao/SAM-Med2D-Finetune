# YOLO Protocol Audit

## Existing Capabilities

- `tools/prepare_yolo_data.py` builds one-class WT boxes from `seg > 0`, uses
  `T1ce/T2/FLAIR` pseudo-RGB and deterministically samples background slices.
- `tools/train_yolo.py` exposes model, epochs, image size, batch, device,
  workers, patience, seed, AMP and save period and writes a resolved run config.
- `tools/evaluate_yolo_recall.py` scans confidence/NMS IoU and reports recall,
  localization IoU and background false positives.
- `infer_volume.py` runs YOLO on the same pseudo-RGB modality order and supports
  top-1 prompt selection.

## Gaps Against the Paper Contract

- Preparation auto-discovers test and can read/export its labels too early.
- One background probability is applied to every split; validation therefore
  may omit most negative slices and understate false positives.
- No dataset manifest binds generated images to `paper_v1` case/slice IDs and
  split SHA-256.
- Training and recall tools do not freeze one detector identity for all SAM
  methods/seeds or persist a reusable per-slice prediction manifest.
- Slice recall ranking alone does not identify the operating point that gives
  the best downstream raw 3D SAM result.

## Chosen Minimal Protocol

Use a single YOLO11m detector at image size 320, fixed 10% padded WT boxes, all
positive training slices, deterministic 1:3 negative sampling, complete-volume
validation, a low-confidence recall scan, and at most two downstream
candidates. Freeze one top-1 operating point and replay its predictions for
every later SAM experiment. Candidate selection is lexicographic so a larger
covering box is never rejected in favor of a tighter box that misses a case or
tumor-positive slice. YOLOv8m remains a documented fallback for a concrete
YOLO11m runtime, memory or convergence failure, not a parallel comparison arm.
