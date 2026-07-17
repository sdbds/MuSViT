# Full-page OMR metric migration report

## Locked identity

- Migration protocol: `canonical_v2_vs_legacy_polish_val_v1`
- Source revision: `a3b659d1e32c78fa76ad76f517a3cf7df74bece5` (clean)
- Checkpoint: `polish_scores_cl_CL-epoch3500.ckpt`
- Checkpoint SHA-256: `ebfbcb14b7bb6fad60280939341c7b603d3713b3d7f67cd9cae3c44b8f161d0c`
- Checkpoint state: `epoch=3399`, `global_step=282200`
- Dataset: `antoniorv6/polish-scores@b3170c8b8f322885b566efe9e264af9328b5603f`
- Split and rows: `val[0:10]`, original order
- Generation: uncached greedy, `maxlen=7512`, 193.924 seconds
- Runtime: RTX 4090 `GPU-a70be80e-9cef-95c2-7557-52448110b38e`, fp16, FlashAttention 2.8.4
- Machine-readable local report: `.cache/full_page_omr/metric-migration-rtx4090.json`

## Aggregate metrics

| Metric | canonical v2 | legacy | v2 - legacy |
| --- | ---: | ---: | ---: |
| CER | 11.408327 | 9.764368 | +1.643959 |
| SER | 13.400844 | 13.473862 | -0.073018 |
| LER | 34.268900 | 34.565119 | -0.296219 |

The CER jump is expected and is not a regression: canonical v2 counts Unicode code points, while the legacy parser treated most ordinary multi-character music tokens as one unit. SER and LER also change because canonical targets exclude EOS. The metric series must therefore use the new `*_v2` names rather than continue legacy history.

All 10 predictions terminated with EOS and none were truncated.

## Per-page deltas

Values are `canonical v2 - legacy`, in percentage points.

| Val row | CER delta | SER delta | LER delta | EOS | Truncated |
| ---: | ---: | ---: | ---: | :---: | :---: |
| 0 | +0.163257 | -0.127109 | -0.481650 | yes | no |
| 1 | +5.003550 | -0.054737 | -0.220126 | yes | no |
| 2 | +3.705803 | -0.092316 | -0.366610 | yes | no |
| 3 | +0.047106 | -0.114758 | -0.565361 | yes | no |
| 4 | -0.993612 | -0.096424 | -0.630800 | yes | no |
| 5 | +1.443326 | -0.052874 | -0.185623 | yes | no |
| 6 | -2.362660 | -0.048791 | -0.224900 | yes | no |
| 7 | +0.646338 | -0.082551 | -0.280201 | yes | no |
| 8 | +1.863519 | -0.064696 | -0.205554 | yes | no |
| 9 | +4.961730 | -0.069562 | -0.262630 | yes | no |
