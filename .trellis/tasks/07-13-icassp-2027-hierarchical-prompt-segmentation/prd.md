# ICASSP 2027: Hierarchical Prompt-Robust Brain Tumor Segmentation

## Working Thesis

Under a frozen automatic-prompt protocol, prompt-jitter training and explicit
hierarchy supervision can reduce SAM-Med2D's raw 3D performance gap between
oracle and automatic boxes without relying on postprocessing or materially
degrading ET segmentation.

This is a falsifiable working thesis, not a result claim. Numerical language
such as "closes 60% of the gap" must not enter the title, abstract, or
conclusion until it is supported by the locked-test results.

## Goal

Convert the existing SAM-Med2D pipeline into a reproducible ICASSP 2027
training-method contribution. The paper must establish whether robustness to
automatic prompts and voxelwise tumor hierarchy can be learned, rather than
attribute gains to detector tuning, prompt heuristics, or 3D postprocessing.

## Audited Baseline Facts

- The current multi-task Adapter was trained with independent class-specific
  ground-truth boxes and reports 0.7560 slice-level validation mean Dice.
- Whole-volume automatic-prompt evaluation reports 0.5104 raw and 0.5460
  postprocessed mean Dice on 167 cases, a raw-to-post gain of 0.0357.
- Those 167 cases come from the validation pool used for early stopping and
  are development evidence, not an independent test result.
- The frozen patient-level split contains 875 train, 187 validation, and 189
  test cases with seed 11171 and no patient overlap.
- Training and inference currently differ in normalization, retained slices,
  prompt source, and prompt dependency. These mismatches must be resolved or
  measured before any method claim.
- Existing hierarchy postprocessing expands TC with ET and WT with TC. It does
  not directly improve ET and must be reported separately from raw output.
- WT continuity is excluded as a primary method: its observed mean Dice gain
  is +0.00143, its bootstrap interval crosses zero, and harmful events exceed
  rescue events.
- The predecessor's approximately 300 training epochs do not by themselves
  establish overfitting. The defensible risks are its small sample, slice-level
  evaluation, repeated validation-driven choices, and opaque checkpoint
  selection. This work addresses those risks directly rather than treating an
  epoch count as evidence.
- Historical 20-train/4-validation experiments are development evidence only
  and must not appear in the paper main table or support a final claim.

## Research Questions and Hypotheses

### RQ1: Where is the automatic-prompt gap introduced?

Using the same A0 checkpoint and cases, decompose performance under full-image,
oracle, controlled-jitter, and frozen-YOLO prompts.

**H1:** Localization and missed-slice errors explain a measurable portion of
the oracle-to-automatic raw 3D Dice gap.

### RQ2: Does prompt-jitter training improve automatic-prompt robustness?

**H2:** A1 improves raw automatic-prompt 3D Dice over A0 under the same frozen
detector predictions and checkpoint-selection rule.

### RQ3: Does hierarchy supervision improve raw nested-label consistency?

**H3:** A2 reduces raw `ET & ~TC` and `TC & ~WT` violations without materially
reducing ET Dice.

### RQ4: Which minimal candidate produces a defensible paper result?

**H4:** At least one of A1, A2, or A3 produces a predeclared accuracy,
prompt-robustness, or difficult-subregion signal on the development protocol;
efficiency may strengthen but cannot replace that evidence. A3 is the planned
combined method, but it is not promoted when a simpler A1 or A2 is stronger
under the same frozen decision contract.

## Claim Boundary

The immediate predecessor, IJCNN 2025 *YOLO-Driven Prompt Generation for
SAM-Based Brain Tumor Segmentation* (`10.1109/IJCNN64981.2025.11228325`),
combines a frozen YOLO locator, sampled point prompts, Adapter tuning, and
label morphology in a small 2D slice study. It does not separately quantify
oracle versus automatic prompts, training/inference prompt mismatch, hierarchy
violations, or postprocessing contribution.

Nested ET/TC/WT constraints are established BraTS methodology: prior work has
used architectural nesting, deterministic Tree-Min-style probability
transformations, structured output parameterizations, and related hierarchy
mechanisms. This paper must not claim invention of hierarchy-aware learning or
of a generic hierarchy loss. Its narrower question is whether an
architecture-preserving, label-preserving soft hinge penalty is sufficient in
the frozen automatic-prompt Adapter setting compared with a canonical existing
hierarchy mechanism.

This work may claim:

1. Patient-level raw 3D measurement of the oracle-to-automatic prompt gap.
2. A controlled PJT and hierarchy-supervision training ablation.
3. Error-source and postprocessing decomposition under a frozen detector.
4. Parameter-efficient automatic-prompt robustness when supported by results.
5. Reduced severe automatic-prompt failures at preserved average accuracy when
   supported by predeclared robustness and bad-case metrics.
6. Non-inferior hierarchy-aware performance from a simpler plug-in penalty
   relative to canonical Tree-Min only when the conditional A4 comparison and
   its frozen uncertainty margin support that statement.

This work must not claim:

- YOLO architecture or detector optimization as a contribution.
- State-of-the-art BraTS segmentation without support from strong baselines.
- A raw model gain when the improvement exists only after postprocessing.
- Independent test performance from the historical 167-case validation run.
- T1ce/T2/FLAIR as the optimal YOLO modality combination, or the four SAM
  modalities as an experimentally optimal subset; modality selection is fixed,
  not a contribution.
- General BraTS state of the art unless a fully comparable formal benchmark
  provides direct evidence.
- Invention of hierarchy loss, Lagrangian optimization, or guaranteed nested
  inference. The proposed term is a soft hinge penalty and may leave residual
  violations.

The preferred claim boundary is the automatic-prompt domain gap, robustness to
detector-generated prompts, and use of ET/TC/WT hierarchy during training. A
small mean gain may support a claim that severe failures are reduced while
average accuracy is preserved only when that outcome was selected through the
predeclared fallback metrics below, not through post-hoc case selection.

### Manuscript Positioning: Slice-Wise vs Volumetric Models

The manuscript must acknowledge established volumetric BraTS systems and then
state the complementary research question: whether a slice-wise medical
foundation model can become robust to imperfect automatic prompts under a
frozen detector protocol. It must not imply that SAM cannot be adapted to 3D,
that prompt quality is already proven to be the sole bottleneck, or that
zero-shot generalization, annotation efficiency, clinical interpretability or
clinical-workflow alignment was established without dedicated evidence.

The conventional 2D U-Net is a matched-dimensionality reference and nnU-Net is
a strong volumetric reference, not a theoretical upper bound. Because these
model families also differ in architecture, pretraining, optimization and
promptability, their comparisons contextualize but do not fully disentangle
foundation-model and dimensionality effects.

Before drafting the abstract, introduction, related work, results narrative,
limitations or reviewer response, apply
`.trellis/spec/guides/paper-positioning-thinking-guide.md` and the task-specific
contract in
`research/manuscript-positioning-2d-vs-3d-2026-07-16.md`.

## Scope and Priority

Priority is assigned by evidence needed for a defensible paper, not by
implementation convenience.

The target is a credible CCF-B ICASSP submission with defensible evidence, not
an A-tier-scale exhaustive benchmark. Extra breadth must not weaken protocol
discipline or delay the central automatic-prompt question.

| Priority | Required work | Cut rule |
| --- | --- | --- |
| P0 | Frozen split and evaluator; A0; prompt-gap decomposition; one-seed A0/A1/A2/A3 development ablation; conditional A4 Tree-Min comparison when hierarchy remains claim-bearing; two-seed full A0/final-candidate confirmation; conventional 2D U-Net; raw/postprocess decomposition; paired statistics and bootstrap confidence intervals; locked test | Cannot be removed without changing the paper claim |
| P1 | nnU-Net v2; third seed; multi-seed confirmation of nonselected ablation arms; expanded failure stratification | Run only after the central result and test protocol are frozen |
| P2 | LoRA comparison; 2.5D consistency; scheduled-sampling cascade; additional prompt or postprocess variants | Exclude from the 60-day critical path |

PJT and the architecture-preserving soft hierarchy penalty are the only primary
method additions. Shared per-volume nonzero-voxel normalization is a protocol
correction applied to all methods. The current Adapter remains the primary PEFT
baseline. Canonical Tree-Min is an existing-method comparator, not a proposed
contribution.

The detector remains `yolo11m.pt` with T1ce/T2/FLAIR pseudo-RGB input and a
recall-first operating point: oversized covering boxes are preferable to
misses. It is trained once, frozen, and replayed for every SAM method and seed.
SAM retains four-channel T1/T1ce/T2/FLAIR input. Neither modality choice is
swept in this paper.

Out of scope for the primary method are threshold grids, detector top-k,
WT-continuity tuning, z-prompt interpolation, web UI changes, and morphology
chains presented as model contributions.

## Frozen Data Protocol

The Kaggle BRaTS 2021 Task 1 source is the only paper dataset. Freeze
`data_brats_paper_v1/split_manifest.json` as `paper_v1`:

| Split | Cases | Permitted use |
| --- | ---: | --- |
| train | 875 | Fit SAM, YOLO, U-Net, and nnU-Net weights only |
| val | 187 | Model selection, detector operating point, ablations, and thresholds |
| test | 189 | One locked final evaluation after all choices are frozen |

A development view may contain 200 train and 43 validation cases sampled only
from the corresponding `paper_v1` partitions with a recorded seed. It is a
compute-saving view, not a fourth experimental population.

The following are protocol invariants:

1. No patient overlap exists across partitions.
2. Test cases are absent from development paths, detector selection, threshold
   scans, and qualitative-case selection before the final freeze record.
3. Training and inference use the same per-volume nonzero-voxel normalization.
4. Every stochastic operation is deterministic for its declared seed.
5. Every run records split, code, checkpoint, configuration, and environment
   identities as defined in `info.md`.
6. Every formal comparison uses the same patient split, frozen YOLO prediction
   files, prompt condition, and evaluator implementation unless the condition
   itself is the explicitly named prompt-decomposition variable.

## Evaluation Contract

All methods export predictions to the same patient-level native-grid 3D
evaluator. Raw and postprocessed results are distinct result families.

### Metric Hierarchy

| Role | Metrics | Use |
| --- | --- | --- |
| Primary decision endpoint | Raw patient-level ET/TC/WT macro-Dice | Checkpoint selection, candidate promotion, paired significance; it cannot be replaced after results are seen |
| Required class results | Raw ET, TC, and WT Dice reported separately | Detect subregion-specific benefit or regression hidden by the macro average |
| Paper main table | Raw ET, TC, WT Dice and HD95 | Final method and baseline comparison |
| Prompt robustness | Oracle-to-YOLO Dice gap; perturbation degradation curve or AUC; worst-quartile Dice; zero-Dice case count; severe-failure rate | Support Fallback B and quantify detector-generated prompt robustness |
| Difficult-subregion guardrails | ET Dice, WT/TC Dice deltas, HD95 | Support Fallback C without hiding compensating regressions |
| Hierarchy diagnostic | `ET outside TC`, `TC outside WT`, total violation voxels, violation-voxel ratio, and violation-case ratio | Explain structure; never sufficient alone because the hierarchy loss directly optimizes it |
| Auxiliary efficiency | Trainable/total parameters, epochs or updates to selected checkpoint, wall-clock time, peak memory, inference cost | Support Fallback D only alongside non-inferior accuracy |
| Secondary output | Postprocessed Dice and its delta from raw | Controlled analysis only; cannot rescue a raw failure or select a checkpoint |
| Internal quality control | Sensitivity, specificity, predicted/GT voxel counts, empty-region status | Debugging and evaluator validation |

HD95 is reported in millimetres using native spacing but does not select a
checkpoint. Undefined empty-surface pairs remain `not_applicable`; cases are
never silently omitted from class-level summaries.

Worst-quartile cases are the fixed 25% of development cases ranked by A0 raw
macro-Dice, then reused for every candidate; each method must not redefine its
own easiest or hardest quartile. A severe failure is initially defined as raw
case macro-Dice below 0.20. Report zero-Dice counts per class and for case
macro-Dice. Freeze the exact perturbation severity grid, AUC integration rule,
quartile membership, severe-failure threshold, and any use of surface Dice on
development before paper-test access. HD95 is the required boundary metric;
surface Dice is optional only if its tolerance is predeclared.
Define the per-case violation-voxel ratio as total unique violating voxels
divided by the predicted ET/TC/WT union, with zero for an empty predicted union;
freeze this denominator convention before comparison.

### Prompt Degradation AUC Contract

The primary robustness AUC is not an area over YOLO confidence thresholds. It
is the normalized area under the raw macro-Dice degradation curve for a fixed
prompt perturbation family. For case `i`, family `f`, and frozen normalized
severity levels `0 = s_0 < ... < s_K = 1`, define:

```text
degradation_i,f(s_k) = raw_macro_dice_i,f(0) - raw_macro_dice_i,f(s_k)
pAUCdeg_i,f = trapezoid_integral(degradation_i,f(s_k), s_k)
```

Lower `pAUCdeg` is better; a negative value is retained when a perturbation
improves a case. Compute it per case before aggregation so candidate-versus-A0
comparisons remain paired. Report separate curves and `pAUCdeg` values for
translation, expansion, contraction, independent-corner coordinate jitter,
and prompt availability. The predeclared summary robustness endpoint is their
unweighted mean after each family's severity axis is normalized to `[0, 1]`;
all family-specific results remain visible. Prompt availability uses fixed,
seeded withholding rates anchored to the observed frozen-YOLO miss and
low-confidence rates. A confidence-threshold scan may diagnose the detector
but cannot replace this AUC or become a fallback endpoint.

For paired method comparisons, use identical case IDs and a two-sided
Wilcoxon signed-rank test on per-case raw Dice. Report per-seed results and
mean +/- standard deviation across seeds. Any additional hypothesis tests
must be labelled secondary rather than promoted after seeing results.
Report case-level paired bootstrap 95% confidence intervals for the primary
raw macro-Dice delta and the selected fallback endpoint; a single decimal-point
difference without paired uncertainty is not sufficient evidence.

### Checkpoint and Overfitting Control

- Save full training/validation curves, selected epoch or update number,
  resolved configuration, checkpoint hash, and selection metadata for every
  formal run.
- Storage policy: keep only the selected `best_model.pth` for SAM Adapter runs
  unless a short diagnostic explicitly requires extra snapshots. Extra epoch
  snapshots are temporary diagnostic artifacts and must not be part of the
  default A0-A3 launch contract.
- Do not use a fixed 300-epoch convention, the final epoch, or the filename
  `best_model.pth` as evidence of optimality. Use early stopping driven by raw
  patient-level 3D development evaluation, or document that the development
  stage is using slice-level `best_model.pth` as a pragmatic storage-limited
  proxy rather than a paper-selected checkpoint.
- Slice-level pooled Dice and loss are health/divergence diagnostics only. They
  may trigger failure investigation but cannot select the paper checkpoint or
  support the main result.
- The current training entry point still writes `best_model.pth` and triggers
  early stopping from slice-level `val_dice_mean`. Until patient-level
  checkpoint selection is implemented, the automatic `best_model.pth` is only a
  development checkpoint. It is acceptable for A0-A3 development iteration under
  the storage-limited policy, but paper claims must label the selection rule
  explicitly or replace it with the common raw 3D evaluator before locked test.
- Paper test is run only after method, checkpoint rule, prompt protocol,
  fallback route, thresholds, and analysis are frozen. Test never tunes a
  method, detector setting, threshold, epoch budget, or checkpoint.

### Gap-Closure Statistic

After results exist, report the descriptive automatic-prompt gap closure as:

```text
gap_closed =
    (candidate_auto_raw - A0_auto_raw)
    / (A0_oracle_raw - A0_auto_raw)
```

Report the numerator and denominator beside the percentage. Do not cap values
or use this statistic when the oracle-to-automatic denominator is non-positive.

## Required Experimental Matrix

### Prompt Decomposition (P0)

Use one frozen A0 checkpoint and identical cases:

| Condition | Purpose |
| --- | --- |
| Oracle class boxes | Training-compatible upper bound |
| Jittered oracle boxes | Controlled translation, scale, and coordinate-noise response |
| Frozen YOLO top-1 box | Automatic end-to-end condition |
| Full-image box / no-useful-prompt condition | Optional diagnosis when missing or low-confidence prompts need a lower reference |

YOLO is a controlled prompt provider. Train it once, freeze one operating
point, persist predictions, and replay the same files for every SAM method and
seed. Its complete contract is owned by
`../07-13-diagnose-prompt-and-postprocess-gap/prd.md`.

The fixed robustness evaluation must independently cover box translation,
scale expansion and contraction, coordinate jitter, and missing/low-confidence
prompt conditions. Use validation YOLO error distributions to define a small
severity grid, then freeze it before candidate comparison. Report the
oracle-to-YOLO gap, degradation curve or normalized AUC, worst-quartile Dice,
zero-Dice count, severe-failure rate, HD95, hierarchy violations, and raw/post
Dice for the predeclared conditions. These diagnostics do not authorize a
YOLO, padding, confidence, or augmentation sweep.

Build the severity grid from the complete 187-case `paper_v1/val` detector
audit, not from the 43-case method-screen subset. Estimate translation and
scale distributions from matched YOLO/oracle positive-slice pairs, but use a
case-balanced empirical distribution: every contributing case has total weight
one, divided equally among its eligible pairs, so large tumors do not dominate.
Missed and low-confidence prompts are summarized separately rather than being
discarded from the matched-pair distribution.

With 187 validation cases, use only coarse nonzero levels at the case-balanced
`q25`, `q50`, `q75`, and `q90` error magnitudes plus the zero-perturbation
reference; do not create finer percentile bins. Report the number of eligible
cases, matched pairs, misses, and a case-bootstrap interval for every quantile.
If all 187 cases are not audited, a perturbation family has too few contributing
cases for stable case-bootstrap quantiles, or adjacent quantiles are not
distinguishable, collapse levels or use a predeclared fixed normalized grid
instead of inventing finer bins. The prompt-gap child task must persist these
decisions as a versioned `severity_grid.json` and point its Y4 output to this
parent AUC contract before G2 closes.

### Training Ablation (P0)

| ID | Shared normalization | Prompt jitter | Hierarchy mechanism | 200/43 single-seed screen | Full paper protocol |
| --- | --- | --- | --- | --- | --- |
| A0 | Yes | No | No | Required baseline | Required, multiple seeds |
| A1 | Yes | Yes | No | Required candidate | Only if selected |
| A2 | Yes | No | Soft hinge penalty | Required candidate | Only if selected |
| A3 | Yes | Yes | Soft hinge penalty | Required candidate | Only if selected |
| A4 | Yes | Match the selected A2/A3 hierarchy-bearing candidate | Canonical HSSN/Tree-Min transform | Conditional existing-method comparator | See A4 decision rule |

All required methods use the same development cases, train budget, automatic
prompt files, selection prompt condition, evaluator, and checkpoint-selection
endpoint. A1, A3, and A4 when matched to A3 may introduce negative-slice prompt
simulation; A0, A2, and A4 when matched to A2 must not silently receive that
treatment.

Screen A0-A3 with one seed on the 200/43 development protocol. Select the
strongest defensible candidate among A1, A2, and A3 using the frozen fallback
gates; the combined A3 is not automatically preferred over a stronger simpler
A1 or A2. Only A0 and that final candidate advance to full `paper_v1` training,
multiple seeds, and locked-test confirmation. Do not run an open-ended loss,
YOLO, modality, or augmentation sweep to manufacture a winner.

### Conditional Tree-Min Comparator (A4)

A4 is triggered after the initial A0-A3 screen and before G3 closes when any
of the following is true:

1. A2 or A3 shows a hierarchy-related accuracy, ET, boundary, or robustness
   signal and hierarchy remains in the intended paper claim.
2. Fallback C is considered for A2/A3 or attributes its result to hierarchy.
3. A3 is the proposed final candidate and the paper argues that the soft
   hierarchy penalty is sufficient or preferable to existing mechanisms.

Do not run A4 when A1 is selected, hierarchy contributes no defensible signal,
and all hierarchy claims are removed. This keeps the comparison hypothesis-
driven rather than adding another method after results are seen.

Before implementation, audit the primary HSSN/Tree-Min paper and official code
when available, and persist the exact label tree, probability transformation,
training loss, inference rule, and citation in
`research/hierarchy-treemin-comparator-audit.md`. A summary-level recreation or
an ad hoc min/max operation must not be labelled Tree-Min.

Run exactly one matched A4 configuration. Use
`a4_treemin_pjt0_dev_seed11171` when comparing with A2 and
`a4_treemin_pjt1_dev_seed11171` when comparing with A3. It uses the same
Adapter, base segmentation loss, PJT state and samples, negative-slice policy,
seed, 200/43 cases, training budget, immutable checkpoints, frozen YOLO files,
and raw patient-level selection endpoint as the hierarchy-bearing candidate.
The only intended difference is:

```text
A2/A3: L_total = L_seg + lambda_hier * mean(
        relu(p_et - p_tc) + relu(p_tc - p_wt)
    )
A4: canonical Tree-Min probability/loss transformation from the audited source
```

Unit tests must verify the audited label-tree semantics, probability ordering,
gradient flow, and equivalence of all non-hierarchy inputs between the matched
A2/A3 candidate and A4.
Report raw macro-Dice, ET/TC/WT Dice, HD95, hierarchy violations, `pAUCdeg`,
worst-quartile Dice, severe failures, trainable parameters, selected
epoch/updates, wall-clock time, and peak memory.

Freeze a matched-candidate-versus-A4 non-inferiority margin at G2 from A0
variability. Apply these outcome rules without post-hoc reinterpretation:

- If the paired confidence interval supports A2/A3 non-inferiority and
  class-level guardrails pass, claim only that the architecture-preserving penalty is
  sufficient under this automatic-prompt protocol, not that it universally
  replaces structured hierarchy methods.
- If A4 materially outperforms the matched A2/A3 candidate, drop the
  soft-penalty sufficiency claim.
  Either include A4 in the formal full-protocol comparison when hierarchy
  remains central, or select A1 and reframe the contribution around PJT and
  prompt robustness.
- If the interval is inconclusive, report the comparison as inconclusive; a
  similar point estimate is not evidence of equivalence or mechanism
  independence.

A4 remains a comparator rather than an A1-A3 proposed candidate. A dev-only A4
may appear in the ablation table when the matched A2/A3 candidate passes the
frozen rule. If A4 is materially stronger and hierarchy remains in the final
claim, A4 must also be run under the formal protocol; it cannot be omitted
because it weakens the preferred narrative.

### Postprocessing Decomposition (P0)

For A0 and the final candidate, report raw output, hierarchy projection only,
morphology/component filtering only, and the full existing postprocessing
chain. The raw table remains the paper headline.

### Comparator Matrix

| Comparator | Priority | Required fairness contract |
| --- | --- | --- |
| Conventional 2D U-Net | P0 | Same cases, modalities, labels, validation-only selection, native-grid evaluator, and raw reporting |
| Canonical HSSN/Tree-Min A4 | P0 when triggered | Match the selected A2/A3 PJT/input/training/evaluation contract; only the audited hierarchy mechanism changes |
| nnU-Net v2 | P1 | Canonical self-configuration, same partitions, no test-derived selection, same exported evaluator |
| Frozen SAM-Med2D | P0 diagnostic | Same automatic prompts and evaluator |
| LoRA | P2 | Same protocol; supplementary only |

The executable baseline protocol is owned by
`../07-13-run-strong-medical-segmentation-baselines/prd.md`. If the final candidate remains
substantially behind nnU-Net, position the result as parameter-efficient
automatic-prompt robustness, not state-of-the-art tumor segmentation.

### Naming Contract

`B0-B5` are workflow stages, `A0-A4` are SAM training/comparator variants, and Fallback
A-D below are evidence/claim routes rather than additional run IDs. B1 remains
the reproducible A0/checkpoint stage, B2 remains the frozen-YOLO prompt-gap
stage, and B3 remains the one-seed A0-A3 method screen plus conditional A4. YOLO keeps its `Y0-Y4`
child-task stages and `y0_yolo11m_paper_v1_seed11171` run identity. The first
development A0-A3 screen also uses seed 11171.
This parent PRD owns candidate promotion, fallback, and paper-claim gates; child
PRDs own implementation details. Any older child wording that promotes only A3
or requires a fixed 1.5-point gain is superseded by this frozen parent decision
contract and must be synchronized before B3 execution.

## Fallback and Decision Gates

The numerical values below are initial development gates, not claims of
universal clinical significance. After B1 quantifies A0 variability on the
fixed development cases using immutable checkpoints and paired bootstrap (and,
if needed, one predeclared repeatability run), freeze the exact margins and
selected fallback endpoints before comparing B3 candidates and before any
paper-test access. This one calibration may adjust the approximate values to
match measured A0 variance; thresholds must not move after candidate results
or test metrics are visible.

### Predeclared Fallback Hierarchy

| Route | Initial go condition on 200/43 development protocol | Permitted paper position |
| --- | --- | --- |
| Fallback A: raw accuracy | Final candidate improves raw patient-level macro-Dice by approximately 1.0 percentage point or more, with no material class regression | PJT and/or hierarchy consistency improves automatic-prompt segmentation |
| Fallback B: prompt robustness | Raw macro-Dice is approximately within +/-0.5 point of A0, while the preselected perturbation AUC/curve, fixed A0 worst-quartile Dice, severe-failure rate, zero-Dice count, or HD95 shows a clear paired improvement | Detector-generated prompt robustness; average accuracy is preserved while severe automatic-prompt failures are reduced |
| Fallback C: difficult subregion | ET or a predeclared difficult-case stratum improves clearly; WT and TC declines are each normally no worse than approximately 0.5 point | Difficult-subregion or hierarchical robustness, with all trade-offs reported |
| Fallback D: auxiliary efficiency | Raw accuracy is non-inferior under a frozen margin and trainable parameters, selected epochs/updates, wall-clock, memory, or compute improves materially | Auxiliary efficiency evidence only; it cannot replace an accuracy or robustness contribution by itself |

Any Fallback A-C route triggers A4 when the selected candidate or intended
claim includes the soft hierarchy penalty. An A1-only accuracy, robustness, or
ET claim does not require a hierarchy comparator.

"Clear improvement" means a consistent paired direction plus a predeclared
effect threshold or paired bootstrap interval that supports improvement; it
does not mean selecting whichever metric happens to improve. A lower hierarchy
violation ratio alone is never a go signal because hierarchy consistency
directly optimizes that measure. A postprocessed-only gain is also never a go
signal for the training method.

If raw Dice, prompt robustness, worst-case/severe-failure outcomes, boundary
quality, difficult-subregion performance, and efficiency all fail their frozen
gates, stop method expansion. Do not enter full-data multi-seed or locked-test
experiments for the candidate.

| Gate | Deadline | Advance condition | Failure action |
| --- | --- | --- | --- |
| G0 Protocol | Day 7 | Split, shared normalization, evaluator, seed control, and artifact contract pass | Stop all performance claims until corrected |
| G1 Prompt provider | Day 14 | Frozen YOLO is replayable and beats full-image prompts under A0 | Diagnose detector/data coordinates once; otherwise use the error-source analysis route without an end-to-end detector claim |
| G2 Gate calibration | Before B3 comparison | Exact A-D margins, robustness endpoint, perturbation grid, worst quartile, severe-failure rule, checkpoint candidates, bootstrap plan, A4 trigger rule, and matched-candidate-vs-A4 non-inferiority margin are frozen from A0 variability | Keep B3 and test blocked |
| G3 Candidate | Day 28 | The simplest strongest A1/A2/A3 candidate passes at least one frozen A-C route; conditional A4 is complete when triggered; D may accompany but not replace A-C | Do not access test; allow one diagnosis/revision cycle, then apply the analysis-only exit gate or stop |
| G4 Baseline/freeze | Day 38 | U-Net is fairly trained; either the final candidate/fallback claim or the analysis-only A0 protocol is selected; checkpoints, thresholds, cases, seeds, and analysis are frozen | Cut P1/P2 and use buffer time |
| G5 Test unlock | Day 39 | Signed freeze record exists and no required P0 experiment is unresolved | Keep test locked |
| G6 Evidence complete | Day 53 | Final statistics, tables, and failure cases are reproducible | Use Days 54-60 only for missing P0 evidence and writing |

No postprocessing, detector retuning, qualitative-case selection, or additional
method component may be used to rescue a failed candidate result after test
access.

### Prompt-Robustness Analysis Protocol

This analysis route is selected when raw-accuracy Fallback A is not met but the
frozen Fallback B or C evidence is credible, or when no candidate passes and
the paper is explicitly reframed as an error-source analysis. It is not
permission to keep searching for a positive method result.

#### Analysis Working Thesis

Under a frozen SAM checkpoint and patient-level 3D protocol, automatic-prompt
degradation can be attributed to measurable localization, missed-slice, and
hierarchy errors, and only part of that degradation is recoverable by training
or postprocessing.

The final causal wording must follow the observed evidence. Do not predeclare
which error source dominates.

#### Analysis P0 Matrix

1. Evaluate full-image, oracle, frozen-YOLO, and controlled-jitter prompts at
   error quantiles derived from validation detector errors.
2. Relate per-case raw Dice loss to translation, scale, coverage, false-positive
   prompts, consecutive missed slices, tumor size, and tumor z-span.
3. Stratify effects by ET/TC/WT and predeclared case-size groups.
4. Decompose hierarchy projection, morphology/component filtering, and the
   full postprocessing chain independently.
5. Report A1/A2/A3 and triggered A4 as negative, mixed, or robustness-specific interventions;
   claim superiority only for the candidate and frozen endpoint that passed.
6. Freeze the analysis variables, strata, figures, and statistical tests on
   validation before one locked-test confirmation.

This route requires no new backbone, detector sweep, or test-derived threshold.

#### If All Method Fallbacks Fail

After the one permitted diagnosis/revision cycle, failure of Fallback A-C is a
method no-go; Fallback D alone does not revive the method claim. Do not run a
full-data candidate, multiple candidate seeds, or a candidate test evaluation.

An analysis-only paper track may continue only when the frozen A0 evidence
still establishes a stable oracle-to-YOLO gap or a reproducible predeclared
perturbation response with paired uncertainty. Its paper structure is fixed as:

1. Motivation: the automatic-prompt domain gap left unresolved by the IJCNN
   predecessor.
2. Protocol: patient-level split, frozen YOLO11m prompts, raw 3D evaluator, and
   checkpoint/test-lock discipline.
3. Measurement: oracle, controlled perturbation, YOLO top-1, and optional
   full-image prompt decomposition with the fixed `pAUCdeg` contract.
4. Error attribution: localization, scale, missing/low-confidence prompts,
   difficult regions, severe failures, HD95, and hierarchy violations.
5. Negative intervention evidence: A1/A2/A3 and triggered A4 development
   results, and why PJT or hierarchy supervision did not yield a defensible
   superiority claim.
6. Postprocessing limits, failure cases, limitations, and implications for
   detector-generated medical prompts.

This track may perform one single-seed, full-protocol A0-only locked-test
confirmation after its analysis variables, strata, figures, and tests are
frozen; it must not select or rescue a candidate on test. If the oracle-to-YOLO
gap and perturbation/error-attribution signals are also unstable or unsupported
by paired uncertainty, terminate the ICASSP submission track and archive the
result as an internal negative study.

## Acceptance Criteria

- [ ] The paper split is immutable, disjoint, hashed, and test-locked.
- [ ] Shared normalization and native-grid evaluator behavior are regression
      tested before baseline claims.
- [ ] Frozen automatic prompts are replayed identically across methods/seeds.
- [ ] A0/A1/A2/A3 development results use the same selection and evaluation
      contract.
- [ ] When hierarchy remains claim-bearing, the A4 trigger is applied before
      G3; its implementation is backed by a primary-source audit and differs
      from A3 only in the hierarchy mechanism.
- [ ] Matched A2/A3-versus-A4 conclusions follow the frozen non-inferiority
      margin and paired uncertainty. A materially stronger A4 is included
      formally or the soft-penalty hierarchy claim is dropped.
- [ ] Raw patient-level 3D macro-Dice is the immutable primary endpoint;
      ET/TC/WT Dice are reported separately and postprocessed Dice remains
      secondary.
- [ ] Formal checkpoints are selected on development raw patient-level 3D
      evaluation from early stopping or a finite immutable candidate set, not
      from pooled slice Dice, fixed epoch 300, or the last checkpoint.
- [ ] Exact Fallback A-D margins, robustness endpoint, perturbation grid,
      difficult-case strata, severe-failure rule, and bootstrap plan are frozen
      from A0 development variability before candidate comparison and test.
- [ ] The robustness endpoint uses the fixed per-case `pAUCdeg` definition,
      complete 187-case validation audit, case-balanced coarse quantiles, and a
      versioned severity-grid artifact rather than confidence-threshold AUC.
- [ ] The simplest strongest candidate among A1/A2/A3 is selected; only A0 and
      that candidate enter full-data multi-seed and locked-test experiments.
- [ ] A training-method claim is rejected when improvement exists only after
      postprocessing or only in directly optimized hierarchy violations.
- [ ] A no-go decision is recorded before full multi-seed work when accuracy,
      robustness, bad-case, boundary, difficult-subregion, and efficiency gates
      all fail.
- [ ] When all method fallbacks fail, the record chooses either the fixed
      analysis-only paper structure with at most one single-seed full-protocol
      A0-only locked-test confirmation or termination of the ICASSP track; it
      does not run a final candidate experiment.
- [ ] The conventional 2D U-Net receives its predefined fair training protocol.
- [ ] Final evidence includes at least two A0 and final-candidate seeds, paired
      statistics, paired bootstrap confidence intervals, prompt decomposition,
      raw/postprocess decomposition, and at least three predeclared
      representative failure cases.
- [ ] The historical 20/4 experiments and 167-case validation results are not
      used as formal paper-test evidence or main-table results.
- [ ] The test set is accessed only after a versioned fallback/claim freeze
      record exists.
- [ ] P1 and P2 work does not delay P0 evidence or the writing buffer.

## 60-Day Schedule

| Days | Deliverable |
| --- | --- |
| 1-7 | Freeze data/evaluator/artifact protocol and reproduce runtime health gates |
| 8-14 | Reproduce A0 and complete prompt-gap decomposition |
| 15-27 | Implement and run one-seed A0/A1/A2/A3 development ablation; run A4 before G3 when triggered |
| 28 | Apply the frozen fallback gates; select A1/A2/A3 candidate, analysis route, or no-go |
| 29-38 | Run conventional U-Net, complete required decompositions, freeze final protocol |
| 39-48 | Run required full seeds and the single locked-test confirmation |
| 49-53 | Complete paired statistics, tables, figures, and failure analysis |
| 54-60 | Slack for missing P0 evidence, paper writing, and reproducibility packaging |

## Document Ownership

- [`info.md`](info.md) owns shared technical design, AutoDL execution gates,
  run identity, artifact layout, and configuration-freeze mechanics.
- [`../07-13-freeze-data-protocol-and-evaluator/prd.md`](../07-13-freeze-data-protocol-and-evaluator/prd.md)
  owns split, preprocessing, and evaluator implementation.
- [`../07-13-diagnose-prompt-and-postprocess-gap/prd.md`](../07-13-diagnose-prompt-and-postprocess-gap/prd.md)
  owns YOLO training/selection and prompt-gap execution.
- [`../07-13-implement-hierarchical-prompt-robust-training/prd.md`](../07-13-implement-hierarchical-prompt-robust-training/prd.md)
  owns PJT, hierarchy loss, A0-A3 implementation, and the conditional A4 comparator.
- [`../07-13-run-strong-medical-segmentation-baselines/prd.md`](../07-13-run-strong-medical-segmentation-baselines/prd.md)
  owns U-Net and nnU-Net execution.
- [`../07-13-finalize-icassp-statistics-and-paper-assets/prd.md`](../07-13-finalize-icassp-statistics-and-paper-assets/prd.md)
  owns locked-test statistics and paper assets.

## Research References

- [`research/literature-survey-sam-medical-2024-2026.md`](research/literature-survey-sam-medical-2024-2026.md)
  provides the broader related-work context.
- [`research/ijcnn2025-predecessor-gap.md`](research/ijcnn2025-predecessor-gap.md)
  records the immediate-predecessor audit and non-overlapping claim boundary.
- [`research/autodl-environment-validation-2026-07-14.md`](research/autodl-environment-validation-2026-07-14.md)
  records the verified training environment and storage constraints.
- [`research/archive/b0-cache-b1-strategy-review-j3s2-2026-07-13.md`](research/archive/b0-cache-b1-strategy-review-j3s2-2026-07-13.md)
  preserves superseded group-server measurements for historical context only.
