# Manuscript Positioning: Slice-Wise Prompt Robustness vs Volumetric Models

Date: 2026-07-16

## Decision Summary

This work is not positioned as an attempt to replace specialized 3D BraTS
segmentation systems. It studies a complementary question:

> Under a frozen and imperfect automatic detector, how much performance is
> lost by a slice-wise medical foundation model, and can prompt-robust and
> hierarchy-aware training reduce that loss?

The research object is therefore the automatic-prompt domain gap in a
slice-wise promptable model. Patient-level raw 3D evaluation measures the
consequences of that design, while volumetric models provide performance
context rather than the primary intervention.

## Introduction Positioning

Use the following logic early in the manuscript:

1. Acknowledge that volumetric CNN and transformer systems, including nnU-Net,
   are established strong approaches for BraTS.
2. State that slice-wise promptable foundation models expose localization as
   an explicit upstream interface. This introduces a measurable train/inference
   prompt shift when oracle training boxes are replaced by automatic boxes.
3. Define the paper question as prompt robustness under a frozen detector, not
   raw competition with specialized volumetric segmentation systems.

Do not write that prompt quality is already known to be the dominant
bottleneck. It is an under-characterized candidate source of degradation until
the full-image, oracle, controlled-jitter and frozen-YOLO decomposition is
complete.

Recommended positioning sentence:

> Rather than competing with specialized volumetric segmentation systems, we
> study a complementary question: how much performance is lost when a
> slice-wise medical foundation model relies on imperfect automatic prompts,
> and whether this loss can be reduced through prompt-robust and
> hierarchy-aware training under a frozen detector protocol.

## Claims and Language Guardrails

Permitted when supported by the frozen experiments:

- Patient-level measurement of the oracle-to-automatic prompt gap.
- Improved robustness to detector-generated prompt errors.
- Reduced severe failures at preserved average raw accuracy.
- Contextual comparison with matched 2D and strong volumetric references.
- Reuse of pretrained representations and an explicit, inspectable prompt
  interface.

Do not claim without dedicated evidence:

- That prompt quality, rather than segmentation capacity or missing volumetric
  context, is the sole or dominant bottleneck.
- That SAM cannot be adapted to 3D. The study intentionally preserves the
  pretrained 2D SAM-Med2D architecture to isolate prompt robustness.
- Zero-shot generalization, annotation efficiency, clinical interpretability,
  or alignment with clinical workflow.
- That nnU-Net is a theoretical or empirical upper bound.
- That a difference between two unmatched model families causally isolates
  dimensionality, pretraining, promptability, or foundation-model effects.
- State of the art from validation-only or non-comparable results.

Use `strong volumetric reference baseline`, not `3D upper bound`.

## Baseline Interpretation Contract

The planned comparisons are useful but do not form a complete factorial
design:

| Method | Dimensionality | Pretrained foundation model | Prompt interface | Role |
| --- | --- | --- | --- | --- |
| 2D U-Net | 2D | No | No | Matched-dimensionality conventional reference |
| SAM-Med2D | 2D | Yes | Yes | Primary promptable model family |
| nnU-Net v2 | Volumetric configuration | No | No | Strong volumetric task reference |

There is no architecture-matched non-pretrained SAM arm and no 3D promptable
foundation-model arm. Consequently, these baselines contextualize the effects
of promptability, pretraining and volumetric context; they do not fully
disentangle them.

Recommended comparison wording:

> We include a matched 2D U-Net and a strong volumetric nnU-Net reference to
> contextualize, rather than fully disentangle, the effects of promptability,
> pretraining, and volumetric context.

Fair comparison requires the same patient split, modalities where applicable,
raw patient-level evaluator and test-lock discipline. Model-specific
preprocessing and optimization must be reported rather than described as
identical. nnU-Net must name the exact configuration and plans used; `nnU-Net
v2` alone is not a complete baseline specification.

## Result-Dependent Narrative

### SAM-Med2D exceeds 2D U-Net but trails nnU-Net

State that the promptable pipeline outperformed the conventional 2D reference
under the evaluated protocol while a gap to the strong volumetric reference
remained. The result is consistent with, but does not isolate, the value of
volumetric context.

### SAM-Med2D trails 2D U-Net

Do not manufacture a competitiveness claim. Continue only when the frozen
prompt decomposition provides stable error attribution or a predeclared
robustness result. Position the work as an analysis of where the automatic
prompt pipeline loses performance and which interventions fail or help.

### SAM-Med2D approaches nnU-Net

Describe it as competitive only after locked-test confirmation, paired
uncertainty and fair evaluator use. Report class-level Dice, HD95, severe
failures, parameters and inference cost; a close validation mean is
insufficient.

## Weak-Detector Decision Path

A weak YOLO detector does not automatically invalidate a prompt-robustness
study. It may serve as a frozen, reproducible error source, but detector
eligibility and end-to-end utility must remain separate claims.

1. Preserve the original strict Y3 result. Never lower a threshold after the
   result and relabel the original run as passing.
2. A documented protocol revision may admit the frozen, zero-fully-missed,
   lexicographically best candidate to the 43-case A0 prompt-gap diagnostic.
3. If frozen YOLO prompts beat full-image prompts and a method passes a frozen
   accuracy or robustness gate, retain the automatic-prompt robustness route.
4. If YOLO does not beat full-image but oracle-to-YOLO degradation and
   controlled perturbation responses are stable, use the error-source analysis
   route without an end-to-end detector claim.
5. If detector utility, prompt-gap attribution and method fallbacks all fail,
   stop candidate expansion and locked-test method claims.

Detector retuning, postprocessing or test-derived thresholds must not rescue a
failed candidate after results are visible.

## Reviewer-Response Template

> We agree that volumetric architectures are established strong approaches for
> BraTS. Our question is complementary: we examine whether a slice-wise medical
> foundation model can become robust to imperfect automatic prompts under a
> frozen detector protocol. We include a conventional 2D U-Net and a strong
> volumetric nnU-Net reference to contextualize the performance of this design,
> while avoiding the stronger claim that these unmatched model families fully
> isolate dimensionality or pretraining effects. Under the common patient-level
> evaluator, the observed result was [insert locked result and uncertainty].

## Manuscript Preflight Checklist

- [ ] Introduction acknowledges strong volumetric BraTS methods before stating
      the complementary prompt-robustness question.
- [ ] The paper says `slice-wise` or `2D architecture preserved by design`, not
      `SAM cannot be 3D`.
- [ ] Prompt quality is described as a hypothesis until decomposition results
      support stronger wording.
- [ ] nnU-Net is called a strong volumetric reference, not an upper bound.
- [ ] Baseline differences are described as contextual evidence, not a complete
      causal decomposition.
- [ ] Zero-shot, annotation-efficiency, workflow and interpretability claims
      are absent unless directly measured.
- [ ] Full-image, oracle, controlled-jitter and frozen-YOLO results use the same
      A0 checkpoint and patient IDs.
- [ ] Raw and postprocessed results remain separate.
- [ ] Result wording follows the frozen fallback route and paired uncertainty.
- [ ] Limitations explicitly include missing volumetric context and incomplete
      factorial isolation of pretraining, promptability and dimensionality.

## Related Project Contracts

- Parent PRD claim boundary and fallback gates: `../prd.md`
- Detector selection and prompt-gap protocol:
  `../../07-13-diagnose-prompt-and-postprocess-gap/prd.md`
- Reusable writing checklist:
  `../../../spec/guides/paper-positioning-thinking-guide.md`

