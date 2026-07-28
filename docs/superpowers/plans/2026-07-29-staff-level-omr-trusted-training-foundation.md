# Staff-level OMR Trusted Training Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first, independently testable slice of the staff-level OMR v2 protocol: deterministic dataset bundles, validated configuration primitives, auditable CTC feasibility, variable-length targets, layered metrics, and a public `prepare-data` command, without changing the legacy training runtime.

**Architecture:** Add a focused `experiments.staff_level_omr.protocol` package whose modules are pure CPU-side foundations and do not import the legacy trainer, Hugging Face, or CUDA. The root CLI exposes `prepare-data` while routing both the explicit `train` subcommand and the omitted-subcommand compatibility form to the unchanged legacy `entrypoint.run`. The second implementation slice will consume these foundations and atomically replace the training runtime; this slice must not emit any `staff_omr_v2` run or checkpoint.

**Tech Stack:** Python 3.11, standard-library dataclasses/JSON/hashlib/decimal/pathlib, Pillow fixtures, PyTorch CPU tensors, `editdistance`, Fire, pytest.

## Global Constraints

- The normative design is `docs/superpowers/specs/2026-07-25-staff-level-omr-trusted-training-protocol-v2-design.md`.
- Use canonical JSON encoded as UTF-8 with sorted object keys, `ensure_ascii=false`, compact separators, no non-finite numbers, and no trailing newline.
- `prepare-data` supports only recursively discovered `*_region.png` / `*_gt.txt` pairs and never guesses `group_id`; `group_regex` must full-match and expose a non-empty named `group_id`.
- Dataset identity is independent of `patch_rows`, `patch_cols`, and CTC exclusions.
- Token parsing is UTF-8 plus Python `str.split()` with no Unicode normalization; tokens sort by UTF-8 bytes; blank id is `0`.
- CTC feasibility is `len(target) + adjacent_repeats <= patch_cols`; no sample is silently filtered.
- Foundation tests require no private data, network, Hugging Face model, or CUDA.
- Do not modify the legacy training behavior, write a temporary v2 checkpoint schema, or claim that `staff_omr_v2` is complete.
- Preserve unrelated dirty worktree changes and stage only files owned by each task.

## File Structure

- `experiments/staff_level_omr/protocol/errors.py`: one typed protocol-validation exception.
- `experiments/staff_level_omr/protocol/canonical.py`: canonical JSON encoding, hashing, loading, and atomic file writes.
- `experiments/staff_level_omr/protocol/config.py`: immutable normalized configuration and early field validation.
- `experiments/staff_level_omr/protocol/vocabulary.py`: vocabulary schema, stable id mapping, encode/decode, and OOV checks.
- `experiments/staff_level_omr/protocol/data_bundle.py`: pair discovery, deterministic group split, bundle generation, bundle cross-validation, and image-verification statistics.
- `experiments/staff_level_omr/protocol/ctc.py`: required-frame analysis, exclusion artifact validation, and train policy application.
- `experiments/staff_level_omr/protocol/batching.py`: variable-length CTC batch collation and exact target reconstruction.
- `experiments/staff_level_omr/protocol/metrics.py`: micro CER and feasible/all capacity-layer aggregation.
- `experiments/staff_level_omr/protocol/__init__.py`: narrow public exports for the future runtime.
- `experiments/staff_level_omr/prepare_data.py`: Fire-friendly public adapter around bundle generation.
- `musvit/cli.py`: staff-level subcommand registry plus one-cycle omitted-`train` compatibility dispatch.
- `tests/test_staff_level_omr_protocol_config.py`: canonical/config unit tests.
- `tests/test_staff_level_omr_data_bundle.py`: generation and validation tests.
- `tests/test_staff_level_omr_ctc.py`: feasibility and exclusions tests.
- `tests/test_staff_level_omr_batching_metrics.py`: variable-target and layered-metric tests.
- `tests/test_staff_level_omr_prepare_data_cli.py`: public command and legacy-routing tests.

---

### Task 1: Canonical JSON And Normalized Configuration

**Files:**
- Create: `experiments/staff_level_omr/protocol/errors.py`
- Create: `experiments/staff_level_omr/protocol/canonical.py`
- Create: `experiments/staff_level_omr/protocol/config.py`
- Create: `experiments/staff_level_omr/protocol/__init__.py`
- Test: `tests/test_staff_level_omr_protocol_config.py`

**Interfaces:**
- Produces: `ProtocolError`, `canonical_json_bytes(value)`, `canonical_sha256(value)`, `read_json(path)`, `write_canonical_json(path, value)`.
- Produces: `StaffOMRConfig.create(..., approved_revisions, default_revisions, revision_resolver=None) -> StaffOMRConfig`.
- Produces: `StaffOMRConfig.to_launch_config() -> dict[str, object]`; `method` is normalized and `input_geometry` is derived.

- [ ] **Step 1: Write failing canonical JSON tests**

```python
def test_canonical_json_is_compact_utf8_and_order_independent():
    left = {"z": "谱", "a": [1, 2]}
    right = {"a": [1, 2], "z": "谱"}
    assert canonical_json_bytes(left) == b'{"a":[1,2],"z":"\\xe8\\xb0\\xb1"}'
    assert canonical_sha256(left) == canonical_sha256(right)


def test_canonical_json_rejects_non_finite_numbers():
    with pytest.raises(ProtocolError, match="finite"):
        canonical_json_bytes({"learning_rate": float("nan")})
```

- [ ] **Step 2: Run the canonical tests and confirm import failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_staff_level_omr_protocol_config.py -q`

Expected: FAIL because `experiments.staff_level_omr.protocol` does not exist.

- [ ] **Step 3: Implement canonical JSON primitives**

Use `json.dumps(..., sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)` and translate serialization/type failures into `ProtocolError` with the offending context. Write bytes directly so files have no trailing newline.

- [ ] **Step 4: Write failing normalized-config tests**

Cover:

```python
config = StaffOMRConfig.create(
    experiment_name="fixture-lora",
    data_path=data_path,
    dataset_bundle_path=bundle_path,
    model_name="musvit",
    method="linear_prob",
    patch_rows=8,
    patch_cols=64,
    start_eval=1000,
    max_epochs=1000,
    approved_revisions={"musvit": {REVISION}},
    default_revisions={"musvit": REVISION},
)
assert config.method == "linear_probe"
assert config.input_geometry == "native_pad"
```

Also assert that `start_eval=0`, `start_eval > max_epochs`, invalid numeric values, invalid experiment names, unsupported methods, explicit `input_geometry`, mutually exclusive revision inputs, unapproved revisions, and `exclude_listed` without an exclusions file fail before model loading.

- [ ] **Step 5: Implement the immutable configuration**

`StaffOMRConfig.create` accepts keyword-only public fields, normalizes `Path` values without embedding resolved locations in training identity, checks readable input paths, derives `native_pad` or `exact_grid`, validates the immutable 40-hex revision against the injected approved registry, and returns a frozen dataclass. `to_launch_config` returns JSON-native values with paths serialized as strings.

- [ ] **Step 6: Run config tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_staff_level_omr_protocol_config.py -q`

Expected: PASS.

- [ ] **Step 7: Commit task 1**

```text
git add experiments/staff_level_omr/protocol tests/test_staff_level_omr_protocol_config.py
git commit -m "feat: add staff OMR protocol configuration"
```

### Task 2: Deterministic Dataset Bundle And Vocabulary

**Files:**
- Create: `experiments/staff_level_omr/protocol/vocabulary.py`
- Create: `experiments/staff_level_omr/protocol/data_bundle.py`
- Test: `tests/test_staff_level_omr_data_bundle.py`

**Interfaces:**
- Consumes: canonical JSON and `ProtocolError` from Task 1.
- Produces: `Vocabulary.from_document(document, manifest_sha256, dataset_id)`, `encode(tokens)`, `decode(ids)`, and `num_classes`.
- Produces: `prepare_dataset_bundle(data_path, dataset_id, group_regex, split_ratios, seed, out) -> PrepareDataReport`.
- Produces: `load_dataset_bundle(bundle_path, data_path, verify_image_hashes="always") -> ValidatedDatasetBundle`.

- [ ] **Step 1: Add failing deterministic-generation tests**

Build six small PNG/text pairs in three or more groups. Assert that two output directories created from the same fixture have byte-identical `bundle.json`, `split_manifest.json`, and `vocabulary.json`; all three splits are non-empty; groups never cross splits; manifest order is stable; vocabulary tokens sort by UTF-8 bytes; and changing no patch-related value is possible because the generator accepts none.

- [ ] **Step 2: Run generation tests and observe missing implementation**

Run: `.venv\Scripts\python.exe -m pytest tests/test_staff_level_omr_data_bundle.py -q`

Expected: FAIL on missing `data_bundle` / `vocabulary`.

- [ ] **Step 3: Implement deterministic discovery and split allocation**

Implement exact final-suffix pairing, normalized `/` relative paths, case-fold collision detection, regex `fullmatch`, named `group_id` validation, `sample_id` derivation, SHA-256 content hashes, exact decimal ratio reduction through `Decimal`/`Fraction`, and group assignment by `SHA256(dataset_id + "\0" + decimal(seed) + "\0" + group_id)` plus largest remainder with `train,val,test` tie order.

- [ ] **Step 4: Implement manifest, vocabulary, index, and bundle output**

Construct the four versioned documents exactly as specified. Write them into a unique sibling temp directory, validate the complete bundle, then rename the directory to `out`; if `out` exists or any step fails, expose no partial final directory and remove only the owned temp directory.

- [ ] **Step 5: Add failing validation tests**

Mutate generated fixtures one condition at a time and assert descriptive rejection for duplicate sample ids, cross-split groups, path escape, target hash mismatch, missing/empty split, bundle/member dataset-id mismatch, cross-hash mismatch, incomplete/extra verification entries, duplicate vocabulary tokens, OOV tokens, and empty targets.

- [ ] **Step 6: Implement full bundle validation**

Validate fixed safe member names, schemas, canonical hashes, dataset-id/source-manifest bindings, unique references, containment after path resolution, readable files, target hashes, vocabulary mapping, verification-index exactness, and the `always`/`cached` image policies. Return immutable samples with parsed tokens and verification statistics.

- [ ] **Step 7: Add cache-policy tests**

Assert:

```python
always = load_dataset_bundle(bundle, data, verify_image_hashes="always")
assert always.image_verification.recomputed == sample_count
assert always.image_verification.status == "content_verified"

cached = load_dataset_bundle(bundle, data, verify_image_hashes="cached")
assert cached.image_verification.hits == sample_count
assert cached.image_verification.status == "cached_metadata"
assert not cached.image_verification.trusted_baseline
```

Then alter an image while preserving size/mtime and confirm `always` fails while the explicitly degraded `cached` mode can only claim metadata verification.

- [ ] **Step 8: Run bundle tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_staff_level_omr_data_bundle.py -q`

Expected: PASS.

- [ ] **Step 9: Commit task 2**

```text
git add experiments/staff_level_omr/protocol tests/test_staff_level_omr_data_bundle.py
git commit -m "feat: add deterministic staff OMR dataset bundles"
```

### Task 3: Auditable CTC Feasibility And Exclusions

**Files:**
- Create: `experiments/staff_level_omr/protocol/ctc.py`
- Test: `tests/test_staff_level_omr_ctc.py`

**Interfaces:**
- Consumes: `BundleSample`, `Vocabulary`, canonical JSON helpers, and `ProtocolError`.
- Produces: `minimum_ctc_frames(target_ids) -> int`.
- Produces: `analyze_ctc_feasibility(samples, vocabulary, patch_cols) -> CTCPreflight`.
- Produces: `apply_train_policy(preflight, manifest_sha256, policy, exclusions_path=None, candidate_path=None) -> CTCPolicyResult`.

- [ ] **Step 1: Write failing required-frame tests**

```python
assert minimum_ctc_frames([1, 2]) == 2
assert minimum_ctc_frames([1, 1]) == 3
assert minimum_ctc_frames([1, 1, 2, 2]) == 6
```

Add train/val/test samples around the `required_frames == patch_cols` boundary and assert only strictly larger values are infeasible.

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_staff_level_omr_ctc.py -q`

Expected: FAIL because `ctc.py` is absent.

- [ ] **Step 3: Implement feasibility records and summaries**

Each sample record contains `sample_id`, split, target length, adjacent repeats, required frames, available frames, and feasibility. The preflight exposes per-split counts and stable count/min/max/mean distributions without modifying the manifest.

- [ ] **Step 4: Write failing policy/exclusions tests**

Assert that:

- `fail` raises with a directly reusable canonical candidate document.
- `exclude_listed` requires exact manifest hash, width, algorithm, sorted unique ids, and exact equality with the actual infeasible train set.
- Missing ids, feasible ids, val/test ids, duplicates, and stale widths fail.
- Excluding every train sample still fails because no retained train sample remains.
- Val/test infeasible samples stay in the preflight result.
- Identical exclusion content at another source path has the same canonical hash.

- [ ] **Step 5: Implement candidate writing and policy validation**

Write candidates only when requested, using the shared canonical writer. Return retained train ids and exclusion hash/count for `exclude_listed`; never filter validation/test. Translate each mismatch into a message naming the bad ids or field.

- [ ] **Step 6: Run CTC tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_staff_level_omr_ctc.py -q`

Expected: PASS.

- [ ] **Step 7: Commit task 3**

```text
git add experiments/staff_level_omr/protocol/ctc.py tests/test_staff_level_omr_ctc.py
git commit -m "feat: add auditable staff OMR CTC preflight"
```

### Task 4: Variable-Length Batching And Layered Metrics

**Files:**
- Create: `experiments/staff_level_omr/protocol/batching.py`
- Create: `experiments/staff_level_omr/protocol/metrics.py`
- Test: `tests/test_staff_level_omr_batching_metrics.py`

**Interfaces:**
- Produces: `CTCBatch` named tuple and `ctc_collate(samples) -> CTCBatch`.
- Produces: `split_concatenated_targets(targets, lengths) -> list[list[int]]`.
- Produces: `input_lengths_for(log_probs) -> torch.LongTensor`.
- Produces: `micro_cer(predictions, targets) -> float`.
- Produces: `layered_metrics(predictions, targets, feasible, feasible_ctc_losses=None) -> LayeredMetrics`.

- [ ] **Step 1: Add failing batching tests**

Create tensors with target lengths 1, 3, and 2. Assert the batch target tensor is one-dimensional concatenation, lengths remain exact, ids/feasibility remain aligned, and reconstructing targets uses lengths rather than searching for blank. Add a final single-sample batch and assert its input-length tensor has shape `(1,)`.

- [ ] **Step 2: Run batching tests and confirm failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_staff_level_omr_batching_metrics.py -q`

Expected: FAIL on missing module.

- [ ] **Step 3: Implement collate and reconstruction validation**

Reject empty batches, non-1D targets, non-positive/mismatched declared lengths, non-stackable images, and concatenated-length inconsistencies. Return CPU `torch.long` lengths and calculate input lengths from runtime `T,N,C`, never from configured batch size.

- [ ] **Step 4: Add failing layered-metric tests**

Use predictions with deliberately different feasible/infeasible errors. Assert micro CER is total edit distance divided by total target length, all-sample CER keeps infeasible samples in the denominator, feasible CER excludes them, and feasible validation loss is the mean of `raw_loss / target_length`. Assert a split with zero feasible samples returns `None` for feasible CER/loss.

- [ ] **Step 5: Implement metrics**

Validate aligned non-empty inputs and finite losses. `LayeredMetrics.to_dict(prefix)` emits the exact future runtime field names such as `val_CER_all`, `val_CER_feasible`, `val_CTC_loss_feasible`, `val_infeasible_samples`, and `val_infeasible_ratio`.

- [ ] **Step 6: Run batching/metric tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_staff_level_omr_batching_metrics.py -q`

Expected: PASS.

- [ ] **Step 7: Commit task 4**

```text
git add experiments/staff_level_omr/protocol/batching.py experiments/staff_level_omr/protocol/metrics.py tests/test_staff_level_omr_batching_metrics.py
git commit -m "feat: add variable staff OMR targets and metrics"
```

### Task 5: Public Prepare-data Command Without A Half-migrated Trainer

**Files:**
- Create: `experiments/staff_level_omr/prepare_data.py`
- Modify: `musvit/cli.py`
- Test: `tests/test_staff_level_omr_prepare_data_cli.py`

**Interfaces:**
- Consumes: `prepare_dataset_bundle` from Task 2.
- Produces: `prepare_data(data_path, dataset_id, group_regex, out, split_ratios=("0.8", "0.1", "0.1"), seed=7) -> dict`.
- Preserves: omitted `musvit staff-level-omr --legacy-options` and explicit `musvit staff-level-omr train --legacy-options` both invoke the current legacy `entrypoint.run`.

- [ ] **Step 1: Write failing CLI routing tests**

Patch the loaded callables and assert:

```text
staff-level-omr prepare-data -> prepare_data
staff-level-omr train        -> legacy run
staff-level-omr --method ... -> legacy run
```

Use a subprocess with a tiny six-pair fixture to run:

```text
.venv\Scripts\python.exe -m musvit.cli staff-level-omr prepare-data ...
```

and assert all four bundle files exist and the process exits zero.

- [ ] **Step 2: Run CLI tests and confirm failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_staff_level_omr_prepare_data_cli.py -q`

Expected: FAIL because the command is not registered.

- [ ] **Step 3: Implement a generic default-subcommand compatibility hook**

Add optional `default_subcommand` metadata to `Experiment`. The staff loader returns `{"train": run, "prepare-data": prepare_data}` and sets `default_subcommand="train"`. In `main`, when the first remaining token is an option (or absent), dispatch directly to the default callable; when it is an explicit registered subcommand, let Fire dispatch the dictionary. Do not special-case training arguments or import `train.py`.

- [ ] **Step 4: Implement the thin prepare-data adapter**

Forward values to `prepare_dataset_bundle`, preserving the tuple of exact ratio strings, and return a JSON-native summary containing output path, dataset id, bundle/manifest/vocabulary hashes, split group/sample counts, and target-length summaries.

- [ ] **Step 5: Run CLI and legacy launcher tests**

Run:

```text
.venv\Scripts\python.exe -m pytest tests/test_staff_level_omr_prepare_data_cli.py tests/test_staff_level_omr_launcher.py -q
```

Expected: PASS with legacy training mocked; no CUDA/model load.

- [ ] **Step 6: Commit task 5**

```text
git add musvit/cli.py experiments/staff_level_omr/prepare_data.py tests/test_staff_level_omr_prepare_data_cli.py
git commit -m "feat: expose staff OMR prepare-data command"
```

### Task 6: Foundation Verification And Boundary Audit

**Files:**
- Modify only if verification exposes a defect in files created by Tasks 1-5.

**Interfaces:**
- Consumes all foundation modules.
- Produces a verified first-slice commit set; it does not produce v2 training artifacts.

- [ ] **Step 1: Run the complete foundation suite**

Run:

```text
.venv\Scripts\python.exe -m pytest tests/test_staff_level_omr_protocol_config.py tests/test_staff_level_omr_data_bundle.py tests/test_staff_level_omr_ctc.py tests/test_staff_level_omr_batching_metrics.py tests/test_staff_level_omr_prepare_data_cli.py tests/test_staff_level_omr_launcher.py -q
```

Expected: PASS.

- [ ] **Step 2: Run all existing staff-level tests and CLI discovery**

Run:

```text
.venv\Scripts\python.exe -m musvit.cli list
.venv\Scripts\python.exe -m pytest tests -q
```

If the full repository suite has pre-existing failures unrelated to this slice, record exact failing tests and prove the focused suite remains green; do not modify unrelated modules.

- [ ] **Step 3: Inspect the diff and protocol boundary**

Run:

```text
git diff --check
git status --short
git log --oneline -8
```

Confirm that:

- the legacy trainer still writes only its legacy checkpoint;
- no module claims protocol version `staff_omr_v2`;
- `prepare-data` is the only newly public runtime path;
- no unrelated dirty file was staged or reverted.

- [ ] **Step 4: Commit verification-only fixes if needed**

Stage only the exact foundation files changed to resolve a verified defect, then commit with a narrowly scoped message. Do not create an empty verification commit.

## Self-Review Record

- **Spec coverage:** This plan covers the first-slice list in §3: configuration normalization, dataset bundle and vocabulary, CTC policy/exclusions, variable target collate, layered metric functions, unit tests, and optionally public `prepare-data`. Input geometry execution, augmentation RNG, model loading, training lifecycle, checkpoint/resume/finalization, README migration, and full CPU training integration remain intentionally assigned to slice 2.
- **Placeholder scan:** No `TBD`, `TODO`, “similar to,” or unspecified error-handling steps remain.
- **Type consistency:** `ValidatedDatasetBundle` supplies immutable samples and `Vocabulary`; `ctc.py` consumes them; batching and metrics are runtime-independent; the public adapter consumes only `prepare_dataset_bundle`.
- **Compatibility check:** Explicit `train` is added only as a routing alias to the current `entrypoint.run`; the omitted-subcommand path remains available, so this slice does not silently switch training semantics.
