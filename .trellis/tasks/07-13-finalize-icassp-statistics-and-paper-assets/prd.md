# Finalize ICASSP Statistics and Paper Assets

## Goal

After a signed Plan A or Plan B freeze, run the required full-data seeds and
single locked-test confirmation, produce paired statistics and paper assets,
and package enough provenance to reproduce every reported number.

## Entry Gate

Do not begin until the parent G3/G4 requirements are satisfied. The freeze
record must specify:

- selected Plan A or Plan B;
- exact methods, checkpoints, seeds, prompt files, thresholds, and postprocess
  conditions;
- test case manifest and evaluator version;
- primary/secondary metrics and hypothesis tests;
- case strata, figure definitions, and failure-case selection rules.

Any prediction-affecting change after unlock invalidates the freeze and must be
handled under the parent `info.md` rerun policy.

## Required Runs

### Plan A

- Full-data A0 and A3 for seeds 11519 and 11520.
- One locked-test evaluation per method/seed using the frozen detector
  prediction set.
- Required conventional U-Net final run under its child PRD.
- Optional third A0/A3 seed 11521 and nnU-Net only after P0 evidence completes.

### Plan B

- The frozen A0 checkpoint and required robustness interventions specified in
  the Plan B matrix.
- One locked-test confirmation of all predeclared prompt conditions, error
  strata, and postprocess conditions.
- Required conventional U-Net context without a proposed-method superiority
  claim.

## Statistical Contract

- Preserve one row per case, method, seed, prompt condition, and postprocess
  condition.
- Primary paired comparison uses two-sided Wilcoxon signed-rank tests on raw
  per-case macro-Dice with identical case IDs.
- Report ET as the co-primary guardrail and ET/TC/WT Dice and HD95 in the main
  results table.
- Report per-seed values and mean +/- standard deviation across seeds.
- Label hierarchy, prompt-error, postprocess, subgroup, and additional metric
  tests as secondary; do not promote them after observing p-values.
- Report effect size, confidence interval where defined, exact sample count,
  missing/undefined metric count, and unadjusted/adjusted status.
- Preserve non-significant and negative findings.

## Paper Asset Contract

Required assets are:

1. Main raw 3D method/baseline table.
2. Prompt decomposition table.
3. A0/A1/A2/A3 development ablation table.
4. Raw versus hierarchy-only, morphology-only, and full-postprocess table.
5. Efficiency/parameter table.
6. Oracle-to-automatic gap figure or Plan B error-attribution figure.
7. At least three failure cases selected by predeclared rules rather than visual
   attractiveness after test results are known.
8. Dataset, run, checkpoint, environment, and artifact provenance appendix.

Every displayed aggregate must be regenerable from archived per-case files by
one versioned script/configuration. Manual spreadsheet edits are not evidence.

## Claim Finalization

Replace the parent working thesis with numerical language only after verifying
the locked outputs. For Plan A, report:

```text
A3 changes automatic-prompt raw 3D macro-Dice from X to Y,
closes G% of the measured oracle-to-automatic gap,
and changes ET Dice by E.
```

For Plan B, state which error sources are associated with degradation and
which interventions recover it only to the extent supported by the frozen
analysis. Avoid causal wording for purely observational per-case associations.

## Acceptance Criteria

- [ ] Test was accessed only after a complete, versioned freeze record.
- [ ] Required Plan A or Plan B runs cover identical locked cases and configs.
- [ ] All main statistics are paired by verified case ID and use raw outputs.
- [ ] Undefined HD95/empty-region cases and exact denominators are visible.
- [ ] Tables and figures regenerate from immutable per-case artifacts.
- [ ] Claims match the selected plan, decision gates, effect sizes, and
      uncertainty; negative findings are retained.
- [ ] P1 omissions are explicitly documented and do not block P0 completion.

## Definition of Done

- Final per-case metrics, statistical outputs, tables, figures, failure cases,
  and reproducibility metadata are archived.
- The paper text contains no historical validation number presented as test.
- The parent acceptance checklist is complete or each exception is documented.
