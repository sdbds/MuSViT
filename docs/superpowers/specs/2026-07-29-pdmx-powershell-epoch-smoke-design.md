# PDMX PowerShell Epoch Smoke Design

## Goal

Make `2.full_page_omr.ps1` run one complete PDMX virtual epoch so the
training stream, both renderers, consumption audit, curated validation, and
checkpoint path are exercised together.

## Frozen Run

The launcher uses:

- `config/Page_OMR_PDMX/pretraining.json`;
- experiment name `pdmx_epoch_smoke`;
- training regime `PDMX`;
- `max_steps=10000`;
- validation every epoch;
- full checkpointing every epoch;
- WSD warmup of 100 steps and decay of 1000 steps;
- eight Windows data workers;
- a noncanonical protocol name dedicated to this smoke run;
- no full-resume checkpoint and no weights-only checkpoint.

The checked-in PDMX config keeps `steps_per_epoch=10000`, so the run reaches
one real virtual-epoch boundary. Lightning performs validation on the final
batch of that epoch.

## Artifact Boundary

The launcher must not download the approximately 96 GB snapshot or generate
manifests implicitly. Before a real run it checks that the PDMX dataset
manifest and vocabulary manifest referenced by the JSON config exist below
`experiments/full_page_omr`.

When an artifact is absent:

- a real run fails before starting training and prints the explicit
  `prepare`, `scan`, and `build-vocabulary` sequence;
- `-DryRun` emits a warning and continues so command construction remains
  testable without the official snapshot.

Snapshot revision and shard integrity remain the Python DataModule's
responsibility.

## Compatibility

The script adds `PDMX` to its accepted training regimes. It does not change
the Python launcher's legacy Arrow behavior, the PDMX production config, or
the machine-local CUDA and Cairo settings.

The PowerShell runtime copy may override only `num_workers`; it must preserve
the PDMX data type, revision, manifest paths, renderer weights, shuffle
buffer, seed, and 10,000-step virtual epoch.

## Verification

`-DryRun` must show:

- the PDMX config-derived runtime file;
- `--finetuning=PDMX`;
- `--max_steps=10000`;
- validation and checkpoint cadence of one epoch;
- the smoke protocol name;
- no `--from_checkpoint` or `--starting_weights`.

The existing PDMX unit suite must remain green. A real run is ready only
after the fixed snapshot, dataset manifest, and vocabulary artifacts exist
locally.
