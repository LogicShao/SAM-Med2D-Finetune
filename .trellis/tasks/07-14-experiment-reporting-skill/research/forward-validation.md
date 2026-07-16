# Record Experiment Report Forward Validation

## Scope

Validate the project skill against a real legacy evaluation artifact and an
incomplete-evidence path without modifying existing experiment outputs.

A separate synthetic fixture exercises the manifest-complete success path at
`research/fixtures/validation_success_fixture_seed11171/`. Its generated report
and unique index row remain under this task and are explicitly non-citable.

## Artifact

```text
outputs/stage7_adapter_verification/fixed20_adapter_baseline/
```

The directory contains `summary_metrics.json`, `summary_metrics.csv`, a
Markdown summary, per-case metadata, predictions, prompt statistics, and
postprocess reports. It does not contain the run-level `manifest.json` required
by the current ICASSP mainline artifact contract.

## Structured Metric Mapping

The aggregate JSON maps cleanly to the report template:

| Report field | Observed value |
| --- | ---: |
| Case count | 20 |
| Raw patient-level macro-Dice | 0.5042245610 |
| Raw ET Dice | 0.3542501077 |
| Raw TC Dice | 0.4811824039 |
| Raw WT Dice | 0.6772411715 |
| Postprocessed macro-Dice | 0.5289547147 |

The validation confirms that raw metrics can lead the report while the higher
postprocessed value remains secondary.

## Incomplete-Evidence Classification

The skill must classify this legacy directory as `incomplete_evidence` unless
other cited files establish the missing run identity, code state, split hash,
seeds, resolved configuration, checkpoint hashes, and terminal exit status.
The correct decision is `insufficient_evidence` or a targeted `investigate`,
not `promote`.

## Manifest-Complete Success Classification

The synthetic fixture contains a terminal manifest and structured raw metrics.
The generated report maps all required reproducibility and primary-result
fields, writes raw results before any secondary analysis, and creates one index
row. Its run status is `succeeded`, while its decision remains
`insufficient_evidence` because no frozen promotion gate exists. This confirms
that execution status and research decision are independent.

## Idempotency Review

The contract requires one report file and one index row per immutable run ID.
Repeated use with unchanged evidence produces no new row. A resumed run updates
the same report only when the manifest confirms that the run ID is unchanged,
and the status transition remains visible in change history.

## Result

The skill structure, metric hierarchy, missing-evidence behavior, and
idempotency rules match the confirmed PRD. No existing report, output, PRD, or
experiment protocol was changed during validation.

The skill validator passed, both fixture JSON files parsed successfully, all
generated report links resolved, and the decision index contained exactly one
row for the fixture run ID.

No `.trellis/spec/` update is required. This reporting contract belongs to the
project-local skill; duplicating it in backend code specs would create a second
source of truth without improving implementation safety.
