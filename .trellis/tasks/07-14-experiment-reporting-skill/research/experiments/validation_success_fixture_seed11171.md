# Experiment Report: `validation_success_fixture_seed11171`

> Synthetic forward-validation output. Do not cite as a research result.

## Verdict

| Field | Value |
| --- | --- |
| Status | `succeeded` |
| Decision | `insufficient_evidence` |
| Profile | `development_screen` |
| Owning task | `.trellis/tasks/07-14-experiment-reporting-skill` |
| Root research task | `.trellis/tasks/07-14-experiment-reporting-skill` |

**Finding:** The fixture completed with raw macro-Dice +0.010 over its fixture baseline, but no frozen promotion gate was declared.

## Reproducibility

| Field | Value | Evidence |
| --- | --- | --- |
| Purpose | Exercise the success reporting path | [`manifest.json`](../fixtures/validation_success_fixture_seed11171/manifest.json) |
| Code revision / dirty state | `synthetic-fixture` / clean | [`manifest.json`](../fixtures/validation_success_fixture_seed11171/manifest.json) |
| Data view / cases | `fixture20` / 20 validation cases | [`manifest.json`](../fixtures/validation_success_fixture_seed11171/manifest.json) |
| Split manifest / SHA-256 | `synthetic-fixture` / `fixture-split-sha256` | [`manifest.json`](../fixtures/validation_success_fixture_seed11171/manifest.json) |
| Method / prompt / postprocess | Adapter / frozen YOLO / disabled | [`manifest.json`](../fixtures/validation_success_fixture_seed11171/manifest.json) |
| Seeds | training 11171; dataset 11171 | [`manifest.json`](../fixtures/validation_success_fixture_seed11171/manifest.json) |
| Command / resolved config | synthetic validation fixture | [`manifest.json`](../fixtures/validation_success_fixture_seed11171/manifest.json) |
| Checkpoints / hashes | fixture base and selected hashes | [`manifest.json`](../fixtures/validation_success_fixture_seed11171/manifest.json) |
| Runtime / environment | 10 s / synthetic fixture | [`manifest.json`](../fixtures/validation_success_fixture_seed11171/manifest.json) |

## Primary Results

| Metric | Candidate | Baseline | Delta | Evidence |
| --- | ---: | ---: | ---: | --- |
| Raw patient-level macro-Dice | 0.510 | 0.500 | +0.010 | [`summary.json`](../fixtures/validation_success_fixture_seed11171/metrics/summary.json) |
| Raw ET Dice | 0.410 | 0.400 | +0.010 | [`summary.json`](../fixtures/validation_success_fixture_seed11171/metrics/summary.json) |
| Raw TC Dice | 0.520 | 0.510 | +0.010 | [`summary.json`](../fixtures/validation_success_fixture_seed11171/metrics/summary.json) |
| Raw WT Dice | 0.600 | 0.590 | +0.010 | [`summary.json`](../fixtures/validation_success_fixture_seed11171/metrics/summary.json) |
| Raw HD95 (mm) | 12.000 | 12.500 | -0.500 | [`summary.json`](../fixtures/validation_success_fixture_seed11171/metrics/summary.json) |

Paired uncertainty/statistics: fixture 95% CI `[+0.002, +0.018]`.

## Guardrails

| Guardrail | Result | Interpretation | Evidence |
| --- | --- | --- | --- |
| Frozen promotion gate | `missing` | A positive delta alone cannot promote a candidate. | `missing` |

Secondary raw/post analysis: not applicable.

## Failures And Missing Evidence

- Failure stage: not applicable.
- Error or interruption: not applicable.
- Last valid evidence: metric summary.
- Missing/conflicting evidence: frozen promotion gate.
- Conclusion limit: this synthetic fixture cannot support a research claim.

## Decision And Next Action

- Decision rationale: no frozen promotion gate can be applied to the fixture.
- Next action: retain this file only as skill forward-validation evidence.

## Artifacts

- Manifest: [`manifest.json`](../fixtures/validation_success_fixture_seed11171/manifest.json)
- Metrics: [`summary.json`](../fixtures/validation_success_fixture_seed11171/metrics/summary.json)
- Logs: missing
- Checkpoints: fixture identities only
- Predictions/plots: missing

## Change History

| Date | Transition | Evidence |
| --- | --- | --- |
| 2026-07-14 | `initial -> succeeded` | [`manifest.json`](../fixtures/validation_success_fixture_seed11171/manifest.json) |
