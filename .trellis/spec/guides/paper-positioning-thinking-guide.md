# Paper Positioning Thinking Guide

Use this guide before drafting an abstract, introduction, related work,
baseline discussion, result narrative, limitations section or reviewer
response for a research task.

## Research-Question Check

- [ ] Is the manuscript framed around the measured research question rather
      than an unsupported broad competition claim?
- [ ] Are established alternatives acknowledged without treating scope as a
      technical impossibility?
- [ ] Are hypotheses clearly separated from observed results?

## Baseline and Causality Check

- [ ] Does each baseline have a named role: matched reference, strong task
      reference, ablation or upper-bound prompt condition?
- [ ] Does the comparison change more than one factor? If so, describe it as
      contextual evidence rather than causal isolation.
- [ ] Is `upper bound` reserved for a defensible bound, not merely a strong
      baseline?
- [ ] Are data split, evaluator, preprocessing, checkpoint selection and
      uncertainty differences disclosed?

## Claim Check

- [ ] Is every claimed advantage directly measured in this study?
- [ ] Are zero-shot, annotation-efficiency, workflow, interpretability and
      clinical claims removed when they are only inherited expectations?
- [ ] Are validation findings kept distinct from locked-test evidence?
- [ ] Does the narrative follow the predeclared fallback route rather than the
      most favorable observed metric?

## Failure-Path Check

- [ ] Is a failed original gate preserved in the record?
- [ ] Is any protocol amendment labeled and frozen before downstream candidate
      or test results?
- [ ] Does the paper have a valid analysis-only or no-go route rather than
      retuning until a method wins?

## Current Project Binding

For the ICASSP 2027 SAM-Med2D task, read the task-specific contract before
writing:

`../../tasks/07-13-icassp-2027-hierarchical-prompt-segmentation/research/manuscript-positioning-2d-vs-3d-2026-07-16.md`

