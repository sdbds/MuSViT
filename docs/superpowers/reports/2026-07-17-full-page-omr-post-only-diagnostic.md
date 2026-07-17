# Full-page OMR post-only checkpoint diagnostic

## Status and identity

- Status: `partial`
- Limitation: no real pre-unfreeze checkpoint was supplied, so no pre/post validation delta is available.
- Source revision: `441771f94a0400156c146853b01dadf586d28e30`
- Checkpoint: `polish_scores_cl_CL-epoch3500.ckpt`
- Checkpoint SHA-256: `ebfbcb14b7bb6fad60280939341c7b603d3713b3d7f67cd9cae3c44b8f161d0c`
- Checkpoint state: `epoch=3399`, `global_step=282200`
- Foundation: `carlospm12/LSMT-MAE-Base-1024-16@eecd5b327521225e65e1c2fe38ab99eb667c1609`
- Foundation encoder digest: `2899556666449187768d28f50412e492981ae97d7b4e7adda191c339b6e94e58`
- Dataset: `antoniorv6/polish-scores@b3170c8b8f322885b566efe9e264af9328b5603f`
- Runtime: locked RTX 4090, fp16, full-prefix greedy, FlashAttention 2.8.4
- Machine-readable local report: `.cache/full_page_omr/post-only-diagnostic-rtx4090.json`

## Validation

| CER_v2 | SER_v2 | LER_v2 | EOS terminated | Truncated | Decode seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 11.408327 | 13.400844 | 34.268900 | 10/10 | 0 | 183.109 |

The report archives all ten per-page metrics and decode times. These values match the independently generated metric-migration report.

## Encoder drift

The comparison domain contains only parameters backed by the fixed ViTMAE snapshot; dynamically initialized, unused `pooler.*` parameters are excluded and recorded as such.

| Scope | Relative L2 |
| --- | ---: |
| Global encoder | 0.252046 |
| encoder.layer.0 | 0.861474 |
| encoder.layer.1 | 0.915315 |
| encoder.layer.2 | 0.910845 |
| encoder.layer.3 | 0.895704 |
| encoder.layer.4 | 0.902140 |
| encoder.layer.5 | 0.910894 |
| encoder.layer.6 | 0.900849 |
| encoder.layer.7 | 0.928860 |
| encoder.layer.8 | 0.955492 |
| encoder.layer.9 | 0.980589 |
| encoder.layer.10 | 1.034785 |
| encoder.layer.11 | 1.040596 |

The magnitude establishes that the encoder moved substantially from the foundation snapshot. It does not establish whether validation improved or degraded after unfreezing; that conclusion remains blocked on a real checkpoint with `global_step < 120000`.
