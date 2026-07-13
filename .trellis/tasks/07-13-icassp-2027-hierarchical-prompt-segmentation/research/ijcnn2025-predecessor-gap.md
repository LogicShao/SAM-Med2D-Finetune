# IJCNN 2025 Predecessor Gap Audit

## Source

- Local PDF: `report/论文/论文/Self-Prompt Segmentation Model for Brain Tumors.pdf`
- Published-paper metadata supplied by the project owner: *YOLO-Driven Prompt
  Generation for SAM-Based Brain Tumor Segmentation*, IJCNN 2025, DOI
  `10.1109/IJCNN64981.2025.11228325`.

## What the PDF Actually Does

1. Uses a YOLO detector to produce a box and samples three point prompts near
   the box center for SAM-Med2D.
2. Fine-tunes SAM-Med2D with Adapter modules.
3. Applies erosion/dilation to labels as a preprocessing choice.
4. Segments tumor core in 256 x 256 2D images; its experiments state that 180
   slices from BraTS 2018 and BraTS 2021 were used.
5. Compares 2D Dice, IoU, sensitivity, and precision with CNN/Transformer
   baselines and includes prompt-count and morphology-iteration ablations.

## Material Gaps

- No patient-level 3D evaluation, HD95, or native-spacing protocol is stated.
- No clearly locked patient-level train/validation/test split is stated.
- No oracle-box, jittered-box, and automatic-YOLO-prompt decomposition is
  reported, so the automatic-prompt error source is not quantified.
- Training robustness to localization error is not a method variable.
- ET/TC/WT nesting is neither trained as an explicit constraint nor reported
  as a violation metric.
- Morphological label processing is treated as a useful component, making it
  essential for a successor to separate raw model output from postprocessing.

## Implication for the ICASSP Submission

The new paper should not claim a better YOLO prompt generator or rely on a
new postprocessing chain. Its narrow, non-overlapping claim is: under frozen
automatic prompts, prompt-jitter training plus hierarchy-aware supervision
improves raw 3D brain-tumor segmentation relative to the same Adapter model
trained with oracle boxes. A paired prompt-condition table and A0/A1/A2/A3
ablation are the shortest credible proof of that claim.

## Minimum Evidence Bar

- Patient-disjoint, frozen split and a held-out test evaluation.
- Raw 3D ET, TC, WT Dice as the headline; postprocessed result reported
  separately.
- Prompt decomposition: oracle, jittered oracle, and YOLO prompts.
- A0/A1/A2/A3 training ablation and a conventional U-Net reference using the
  same evaluator.
- At least two final seeds, paired per-case testing, and representative
  failures. A third seed and nnU-Net are strengthening evidence, not blockers.
