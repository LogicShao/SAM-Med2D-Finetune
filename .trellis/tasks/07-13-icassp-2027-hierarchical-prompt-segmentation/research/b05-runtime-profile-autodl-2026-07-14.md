# B0.5 AutoDL Runtime Profile

## Scope

This note records the one-time AutoDL runtime qualification profile required by
`info.md` before A0. It supersedes all retired-server throughput measurements.

## Fixed Conditions

- Environment: AutoDL `RTX 4090D`, `cuda:0`, PyTorch `2.6.0+cu124`
- Seed and dataset seed: `11171`
- Data view: `paper_dev_v1` train/val
- Cache root: `/root/autodl-tmp/cache/paper_dev_v1`
- Cache policy: `cache_max_cases=8`
- Prompt policy: positive slices only, `negative_to_positive_ratio=0`,
  `negative_prompt_box=zero`
- Optimizer path: Adapter finetuning, AMP off, cuDNN enabled,
  `cudnn_benchmark=false`
- Profile mode: `epochs=1`, `max_train_steps=100`, `val_subset_size=4`,
  `profile_performance=true`, `profile_stages=true`,
  `profile_gpu_utilization=true`

The valid evidence set is the `_r2` run family. An earlier partial attempt
stopped before training because the synchronized AutoDL code directory is not a
Git clone; those partial artifacts are not used for selection.

## Result Table

Summary CSV:
`/root/autodl-tmp/runs/b05_profile_matrix_seed11171_r2.csv`

| Run ID | Batch | Workers | Samples/s | Peak alloc MiB | Data wait s | H2D s | GPU compute s | GPU util % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `b05_profile_b8_w2_seed11171_r2` | 8 | 2 | 37.2928 | 4598.8013 | 0.0046 | 0.0012 | 0.2062 | 47.0789 |
| `b05_profile_b8_w4_seed11171_r2` | 8 | 4 | 37.6740 | 4599.1763 | 0.0048 | 0.0013 | 0.2034 | 44.6486 |
| `b05_profile_b8_w8_seed11171_r2` | 8 | 8 | 34.7334 | 4598.8013 | 0.0069 | 0.0013 | 0.2193 | 39.2000 |
| `b05_profile_b12_w2_seed11171_r2` | 12 | 2 | 52.2536 | 5653.3701 | 0.0049 | 0.0018 | 0.2200 | 53.4000 |
| `b05_profile_b12_w4_seed11171_r2` | 12 | 4 | 50.6736 | 5650.1826 | 0.0059 | 0.0019 | 0.2262 | 48.6829 |
| `b05_profile_b12_w8_seed11171_r2` | 12 | 8 | 44.5028 | 5653.3701 | 0.0101 | 0.0019 | 0.2542 | 57.3830 |
| `b05_profile_b16_w2_seed11171_r2` | 16 | 2 | 62.1502 | 6677.6416 | 0.0056 | 0.0024 | 0.2465 | 61.9111 |
| `b05_profile_b16_w4_seed11171_r2` | 16 | 4 | 59.5297 | 6678.3916 | 0.0064 | 0.0024 | 0.2572 | 69.8723 |
| `b05_profile_b16_w8_seed11171_r2` | 16 | 8 | 61.8127 | 6678.3916 | 0.0094 | 0.0024 | 0.2443 | 63.6889 |

## Decision

Freeze the runtime candidate for the next gate as:

- `batch_size=16`
- `num_workers=2`
- AMP off
- cuDNN enabled
- `cudnn_benchmark=false`
- `cache_max_cases=8`
- `persistent_workers=false`
- `non_blocking_transfer=false`

Rationale:

1. `b16/w2` is the fastest measured configuration.
2. `b16/w8` is statistically too close to justify the extra worker count and
   has higher batch-wait overhead.
3. All runs are far below the 22 GiB peak-allocation gate; memory is not the
   limiting factor on AutoDL.
4. CPU batch wait and H2D copy are both tiny relative to GPU compute, so the
   current bottleneck is the compute graph rather than dataset I/O.

## Interpretation

- Throughput increases monotonically with batch size over the profiled range.
- Worker counts above `2` do not help on AutoDL for this cached dataset path
  and can reduce throughput.
- The GPU is not saturated by this configuration family, but the parent PRD
  limited the one-time infrastructure sweep to batch sizes `8/12/16`. Any
  larger-batch experiment would be a new method/runtime decision and must not
  be silently substituted for A0.

## Next Gate

Run the required 3-epoch stability check on AutoDL with the frozen
`batch_size=16`, `num_workers=2` configuration before any A0 evidence run.
