# Staff-level OMR trusted training protocol v2

This experiment trains a MuSViT encoder, a two-layer bidirectional LSTM, and a
CTC head on cropped staff images. Version 2 is an auditable baseline: it fixes
the identities of the data, vocabulary, augmentation, input geometry, base
weights, optimizer, and checkpoints. It does **not** claim better CER or higher
throughput than the legacy trainer.

The executable v1 path has been removed. Legacy state-dict-only checkpoints
cannot be resumed by v2.

## 1. Prepare a dataset bundle

The source directory may be nested, but every image must have a sibling target:

```text
score-001/page-01_staff-01_region.png
score-001/page-01_staff-01_gt.txt
```

Targets are UTF-8 text split with Python `str.split()`. First generate the
manifest, group-disjoint train/validation/test split, closed-corpus vocabulary,
and image verification index:

```bash
uv run --frozen musvit staff-level-omr prepare-data \
  --data_path D:/datasets/staff-omr \
  --dataset_id catedrales-v1 \
  --group_regex '(?P<group_id>score-[^/]+)/.*_region[.]png' \
  --split_ratios 0.8 0.1 0.1 \
  --seed 7 \
  --out D:/datasets/staff-omr-bundle
```

`group_regex` is a full match against each POSIX-style relative image path and
must contain a named `group_id` capture. Choose a group that must never cross
splits, normally the work or page identity. The command refuses unmatched files,
fewer than three groups, an existing output directory, broken pairs, or empty
targets. It publishes these files atomically:

```text
bundle.json
split_manifest.json
vocabulary.json
image_verification_index.json
```

The vocabulary intentionally covers the complete closed corpus, including
validation and test tokens. This avoids undefined output ids; it is not an
open-vocabulary evaluation protocol.

## 2. Start training

The three user-owned identity/location fields are required:

```bash
uv run --frozen musvit staff-level-omr train \
  --experiment_name catedrales \
  --data_path D:/datasets/staff-omr \
  --dataset_bundle_path D:/datasets/staff-omr-bundle \
  --model_name musvit \
  --method lora \
  --patch_rows 8 \
  --patch_cols 128 \
  --batch_size 8 \
  --num_workers 6 \
  --learning_rate 0.0003 \
  --max_epochs 1000 \
  --start_eval 20 \
  --patience 30 \
  --device cuda \
  --verify_image_hashes always
```

Omitting the `train` word remains a one-release compatibility alias. The
deprecated spelling `--method linear_prob` normalizes to `linear_probe`, and
`--shape_patches='[8,128]'` normalizes to
`--patch_rows 8 --patch_cols 128`.
Canonical commands and recorded configs use only `linear_probe`, `patch_rows`,
and `patch_cols`. Old and new patch arguments cannot be mixed.

Important options:

| Option | Default | Contract |
|---|---:|---|
| `model_name` | `musvit` | `musvit` or `musvit_light` |
| `model_revision` | registry default | Approved immutable 40-character commit SHA |
| `resolve_model_revision` | false | Resolve the Hub head once and accept it only if already approved |
| `method` | `lora` | `linear_probe` or `lora` |
| `patch_rows`, `patch_cols` | `8`, `64` | Positive patch-grid dimensions |
| `augmentation_profile` | `staff_omr_train_v1` | Exact declared profile, or `none` |
| `train_infeasible_policy` | `fail` | `fail` or `exclude_listed` |
| `batch_size` | `8` | Positive integer |
| `num_workers` | `6` | Non-negative; does not change sample order or augmentation |
| `learning_rate` | `0.0003` | Adam learning rate |
| `max_epochs` | `1000` | Positive epoch budget |
| `start_eval` | `20` | Must be between 1 and `max_epochs` |
| `patience` | `30` | Validation evaluations without strict improvement |
| `seed` | `7` | Non-negative base seed |
| `device` | `cuda` | `cpu`, `cuda`, or `auto` |
| `verify_image_hashes` | `always` | `always` or explicitly degraded `cached` |

The approved revisions and weight SHA-256 values live in
`protocol/backbone.py`. Model inspection validates Hub tree metadata and the
downloaded weight contents before training.

## 3. Input geometry and CTC capacity

The method uniquely determines geometry:

| Method | Geometry | Width rule |
|---|---|---|
| `linear_probe` | `native_pad` | `patch_cols` must equal the backbone's native columns |
| `lora` | `exact_grid` | compatible `patch_cols` values are allowed |

`linear_probe` keeps the frozen encoder on its pretrained positional grid and
bottom-pads white pixels. `lora` resizes to the requested grid and verifies
position-embedding interpolation against the explicit bicubic reference. These
geometries are not directly configurable. Their v2 results must not be mixed
with legacy CER series.

For a target `y`, CTC needs:

```text
minimum_frames(y) = len(y) + adjacent_equal_pairs(y)
```

The default `fail` policy rejects any infeasible training sample before model
weights are loaded. The failed run contains
`train_exclusions.candidate.json`. Recovery is explicit:

1. Prefer LoRA with a larger `patch_cols` when the time axis is genuinely too
   short.
2. Otherwise review the candidate file.
3. Start a **new** run with
   `--train_infeasible_policy exclude_listed` and
   `--train_exclusions_path <reviewed-candidate>`.

Linear probing cannot gain time steps by increasing `patch_cols`; switch to
LoRA or explicitly exclude the exact infeasible set. Different widths or
exclusion sets define different training contracts, so compare them with the
reported capacity statistics rather than treating their CER values as the same
population.

Validation and test retain infeasible samples. Every evaluation records:

- `*_CER_all`: micro CER over every sample; this selects `best.pt` for validation.
- `*_CER_feasible`: micro CER over samples that fit the time axis.
- `val_CTC_loss_feasible`: CTC loss over only feasible validation samples.
- `*_capacity`: feasible/infeasible counts plus target-length and required-frame
  distributions.

CTC uses blank id `0`, real tokens start at id `1`, and
`num_classes = len(vocabulary.tokens) + 1`. Loss uses
`zero_infinity=False`; non-finite loss or gradients terminate the current epoch
instead of silently discarding data.

## 4. Resume

Resume uses only the same run's committed `checkpoints/last.pt`:

```bash
uv run --frozen musvit staff-level-omr resume \
  D:/runs/catedrales/<run-directory> \
  --max_epochs 1200 \
  --num_workers 2
```

Allowed operational overrides are `max_epochs` (increase only), `num_workers`,
`data_path`, `device`, and `verify_image_hashes`. Package patch/build or runtime
environment drift requires `--allow_env_drift` and is recorded. Core package
major/minor drift, protocol/data/model/geometry changes, optimizer-name
mismatches, cross-run checkpoints, and legacy checkpoints are rejected.

Resume is exact at complete epoch boundaries in the supported CPU protocol
test. It does not promise mid-batch, cross-device, GPU-bitwise, or
cross-dependency-version replay. `verify_image_hashes=always` re-reads every
image on every new process, including resume; `cached` is faster but marks the
run as a degraded, non-baseline verification mode.

## 5. Run artifacts

New runs never reuse a directory:

```text
<output_root>/<experiment_name>/<UTC>-<contract-hash>-<run-id>/
  run.json
  dataset_bundle.json
  split_manifest.json
  vocabulary.json
  image_verification_index.json
  train_exclusions.json
  metrics.jsonl
  checkpoints/
    last.pt
    best.pt
  test.json
  summary.json
```

Only applicable exclusion files are present. `last.pt` is the epoch transaction
commit point; sidecars are repaired from it after an interrupted write.
`best.pt` is selected by strictly lower `val_CER_all`. Final test runs from
`best.pt`, and finalization is idempotent.

Checkpoints save trainable state, optimizer state, exact parameter-name mapping,
the full vocabulary, input/training contracts, manifest identity, and approved
base-weight identity. They omit the large immutable backbone and complete
manifest. A standalone checkpoint can decode already-produced ids without
scanning target files, but inference still needs the recorded base revision and
matching weight SHA-256.

Structured JSON artifacts are authoritative; terminal output is only a
convenience view.

## 6. Windows launcher

The repository launcher requires both the raw data and prepared bundle:

```powershell
.\4.staff_level_omr.ps1 `
  -DataPath D:\datasets\staff-omr `
  -DatasetBundlePath D:\datasets\staff-omr-bundle
```

Add `-DryRun` to validate inputs and print the exact canonical v2 command
without loading a model.
