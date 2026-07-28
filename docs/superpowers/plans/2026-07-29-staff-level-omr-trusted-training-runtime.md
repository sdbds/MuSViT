# Staff-Level OMR Trusted Training Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the executable staff-level OMR v1 path with one auditable v2 runtime that prepares deterministic inputs, trains or resumes at epoch boundaries, and emits self-contained run artifacts accepted by every requirement in design spec section 18.

**Architecture:** Keep the already-implemented protocol foundation as the authority for canonical JSON, dataset bundles, vocabulary, CTC feasibility, collation, and layered metrics. Add focused modules for deterministic seeds and augmentation, input geometry, the pinned backbone registry, the task model, data loading, run artifacts, checkpoints, and lifecycle orchestration. Public CLI modules become thin adapters to this runtime; there is no second legacy training implementation.

**Tech Stack:** Python 3.11, PyTorch 2.13, torchvision 0.28, Transformers 4.57, PEFT 0.19, Albumentations 2.0.8, Pillow, Hugging Face Hub, Fire, pytest.

## Global Constraints

- Protocol identity is exactly `staff_omr_v2`; v1 checkpoints are rejected rather than upgraded implicitly.
- Canonical methods are `linear_probe` and `lora`; `linear_prob` remains a one-cycle input alias only.
- Canonical patch arguments are `patch_rows` and `patch_cols`; `shape_patches` remains a one-cycle alias and cannot be combined with either canonical argument.
- `linear_probe` derives `native_pad`; `lora` derives `exact_grid`; callers cannot set input geometry.
- CTC blank is id `0`; vocabulary token index `i` maps to id `i + 1`; `num_classes = len(tokens) + 1`.
- CTC loss uses `zero_infinity=False`; every retained training target must satisfy `minimum_ctc_frames(target) <= patch_cols`.
- The augmentation contract is handwritten, contains normalized effective values, and never derives identity from `Albumentations.to_dict()`.
- Semantic sample order and augmentation are independent of `num_workers`; worker assignment is operational metadata.
- `DataLoader` capability is determined by the presence of `in_order` in its signature; the torch version is recorded but is not a substitute for the capability probe.
- The production backbone loader uses `ViTModel.from_pretrained(..., trust_remote_code=False, add_pooling_layer=False, output_loading_info=True)` against an approved immutable revision.
- Runtime geometry contains no literal `64 x 64` reshape and no literal `1024` padding assumption.
- Checkpoint state is committed only at epoch boundaries; `last.pt` is the transaction commit point.
- A checkpoint embeds the vocabulary and manifest hash, not the complete manifest.
- Resume occurs only from the same run's `last.pt`; environment drift and max-epoch extension follow the hard/soft rules in spec section 12.3.
- Tests use local tiny or dummy backbones and never require a network or CUDA device.
- Windows tests prepend `D:\BaiduNetdisk\module\ImageViewer` to `PATH` when the whole repository imports CairoSVG.

---

### Task 1: Deterministic Seeds And Normalized Augmentation

**Files:**
- Create: `experiments/staff_level_omr/protocol/seeding.py`
- Create: `experiments/staff_level_omr/protocol/augmentation.py`
- Create: `tests/test_staff_level_omr_augmentation.py`

**Interfaces:**
- Produces: `seed_digest(label: str, *parts: object) -> bytes`, `seed32(...) -> int`, `seed256(...) -> int`, `sample_order_key(seed: int, epoch: int, sample_id: str) -> tuple[bytes, bytes]`, `reset_epoch_rng(seed: int, epoch: int) -> None`, and `worker_base_seed(seed: int, epoch: int) -> int`.
- Produces: `augmentation_contract(profile: str) -> dict[str, object]`, `build_augmentation(profile: str) -> albumentations.Compose | None`, and `preflight_augmentation(profile: str) -> dict[str, object]`.
- Consumes: `ProtocolError` and canonical hashing from the foundation.

- [ ] **Step 1: Write failing seed and profile tests**

```python
def test_seed_digest_uses_protocol_domain_and_all_256_bits():
    low = seed256("augment", 7, 1, "sample")
    high = low ^ (1 << 200)
    assert low != high
    assert seed32("augment", 7, 1, "sample") == (
        seed_digest("augment", 7, 1, "sample")[:4]
    )

def test_staff_profile_declares_only_effective_odd_blur_kernels():
    contract = augmentation_contract("staff_omr_train_v1")
    assert contract["transforms"][6]["transforms"][0]["transforms"][0][
        "blur_limit"
    ] == [3, 3]
    assert contract["transforms"][6]["transforms"][0]["transforms"][1][
        "blur_limit"
    ] == [5, 5]
```

- [ ] **Step 2: Run the focused tests and confirm missing-module failures**

Run: `.venv\Scripts\python.exe -m pytest tests/test_staff_level_omr_augmentation.py -q`

Expected: collection fails because `protocol.seeding` and `protocol.augmentation` do not exist.

- [ ] **Step 3: Implement exact seed domains and the declarative profile**

```python
def seed_digest(label: str, *parts: object) -> bytes:
    payload = b"\0".join(str(part).encode("utf-8") for part in parts)
    return hashlib.sha256(
        b"staff_omr_v2\0" + label.encode("utf-8") + b"\0" + payload
    ).digest()

def seed32(label: str, *parts: object) -> int:
    return int.from_bytes(seed_digest(label, *parts)[:4], "big")

def seed256(label: str, *parts: object) -> int:
    return int.from_bytes(seed_digest(label, *parts), "big")
```

Build every Albumentations transform with every parameter in the design spec supplied explicitly. Validate normalized constructor attributes, assert every probed leaf has `p > 0`, reject warnings, capture actual OpenCV Gaussian and motion-kernel sizes across the fixed seed matrix, and probe both same-seed equality and a bit-200-only seed difference.

- [ ] **Step 4: Run deterministic augmentation tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_staff_level_omr_augmentation.py -q`

Expected: all tests pass without Albumentations warnings.

- [ ] **Step 5: Commit the seed and augmentation contract**

Commit exact files with message: `feat: add deterministic staff OMR augmentation`

---

### Task 2: Input Geometry And Backbone Registry

**Files:**
- Create: `experiments/staff_level_omr/protocol/geometry.py`
- Create: `experiments/staff_level_omr/protocol/backbone.py`
- Create: `tests/test_staff_level_omr_geometry.py`
- Create: `tests/test_staff_level_omr_backbone.py`

**Interfaces:**
- Produces: immutable `BackboneMetadata`, `GeometryPlan`, and `BackboneLoadResult`.
- Produces: `APPROVED_BACKBONES`, `approved_revisions()`, `default_revisions()`, `inspect_backbone(...)`, `load_backbone(...)`, and `reference_interpolate_pos_encoding(...)`.
- Produces: `build_geometry_plan(metadata, method, patch_rows, patch_cols)`, `StaffImageProcessor(plan)`, and `extract_spatial_grid(last_hidden_state, plan)`.

- [ ] **Step 1: Write failing registry, processor, and interpolation tests**

```python
def test_registry_pins_musvit_weight_identity():
    entry = APPROVED_BACKBONES["musvit"]
    assert entry.revision == "0e91c7b223b4da30f259198c92045d0cb90e3f2e"
    assert entry.weight_sha256 == (
        "109bbaf31d9f2184df1b841579e06d25bc58ed6a42a10dd5f4a5d27d01889db2"
    )

def test_native_pad_resizes_content_then_bottom_pads_white():
    plan = build_geometry_plan(META, "linear_probe", 8, 64)
    tensor = StaffImageProcessor(plan)(image)
    assert tensor.shape == (3, META.image_height, META.image_width)
    assert torch.all(tensor[:, plan.content_height :, :] == 1)

def test_exact_grid_uses_transformers_equivalent_bicubic_interpolation():
    actual = embeddings.interpolate_pos_encoding(tokens, height, width)
    expected = reference_interpolate_pos_encoding(
        embeddings.position_embeddings, height, width, META
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
```

- [ ] **Step 2: Run the geometry and backbone tests and confirm failures**

Run: `.venv\Scripts\python.exe -m pytest tests/test_staff_level_omr_geometry.py tests/test_staff_level_omr_backbone.py -q`

Expected: collection fails on missing modules.

- [ ] **Step 3: Implement metadata validation and geometry**

Validate the raw config before model construction: `model_type == "vit_mae"`, `architectures` contains `ViTMAEForPreTraining`, `num_channels == 3`, all dimensions are positive, and native dimensions divide by patch dimensions. The processor uses Pillow RGB conversion plus torchvision bilinear resize with `antialias=True`, produces `[0, 1]` tensors without mean/std normalization, and applies only bottom white padding in `native_pad`.

- [ ] **Step 4: Implement immutable model evidence and fixed loading**

Verify README/config git-blob identities, absence of `preprocessor_config.json`, and the actual safetensors size and SHA-256 before calling:

```python
model, loading_info = ViTModel.from_pretrained(
    entry.model_id,
    revision=entry.revision,
    trust_remote_code=False,
    add_pooling_layer=False,
    output_loading_info=True,
)
```

Reject missing or mismatched keys and reject unexpected keys except `decoder.*`. Compare loaded config with the inspected raw config. Run the installed Transformers positional interpolation against the explicit bicubic reference (`align_corners=False`, `antialias=False`).

- [ ] **Step 5: Run focused tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_staff_level_omr_geometry.py tests/test_staff_level_omr_backbone.py -q`

Expected: all tests pass using local tiny fixtures; no network is contacted.

- [ ] **Step 6: Commit geometry and registry**

Commit exact files with message: `feat: pin staff OMR backbone geometry`

---

### Task 3: Task Head, LoRA, And Optimizer Contract

**Files:**
- Create: `experiments/staff_level_omr/protocol/modeling.py`
- Create: `experiments/staff_level_omr/protocol/optimization.py`
- Create: `tests/test_staff_level_omr_model_v2.py`

**Interfaces:**
- Produces: `StaffOMRModel`, `build_model(backbone, metadata, config, num_classes)`, `task_head_contract(metadata, num_classes)`, `greedy_ctc_decode(log_probs)`, `trainable_state_dict(model)`, `build_optimizer(model, learning_rate)`, and `optimizer_parameter_names(model)`.
- Consumes: geometry grid extraction, deterministic initialization seed, and normalized `StaffOMRConfig`.

- [ ] **Step 1: Write failing model contract tests**

```python
def test_task_head_exact_shape_and_keys():
    model = build_model(dummy_backbone, META, config, num_classes=9)
    assert model.projection.weight.shape == (256, META.hidden_size)
    assert model.rnn.num_layers == 2
    assert model.rnn.bidirectional
    assert model.classifier.out_features == 9
    assert set(task_head_contract(META, 9)) == EXPECTED_CONTRACT_KEYS

def test_optimizer_has_one_utf8_sorted_trainable_group():
    optimizer, names = build_optimizer(model, 3e-4)
    assert names == sorted(names, key=lambda value: value.encode("utf-8"))
    assert len(optimizer.param_groups) == 1
    assert all(parameter.requires_grad for parameter in optimizer.param_groups[0]["params"])
```

- [ ] **Step 2: Run model tests and confirm missing-module failures**

Run: `.venv\Scripts\python.exe -m pytest tests/test_staff_level_omr_model_v2.py -q`

- [ ] **Step 3: Implement the exact head and trainability policy**

The forward path is projection (`hidden_size -> 256`, no bias), mean over rows, two-layer bidirectional LSTM (`256`, dropout `0.5`), classifier (`512 -> num_classes`), and log-softmax. Allocate zero recurrent states from the projected tensor's device and dtype. For `linear_probe`, freeze the full backbone. For `lora`, inject PEFT targets `query`, `key`, and `value` with `r=8`, `alpha=16`, `dropout=0.1`, `bias="none"`, and `use_rslora=True`, then initialize the complete task head after resetting the protocol init seed.

- [ ] **Step 4: Implement exact Adam construction**

Create one parameter group from UTF-8-sorted fully qualified trainable names with Adam `betas=(0.9, 0.999)`, `eps=1e-8`, `weight_decay=0`, and no scheduler. Persist the ordered name list for checkpoint validation.

- [ ] **Step 5: Run focused tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_staff_level_omr_model_v2.py -q`

- [ ] **Step 6: Commit model and optimizer**

Commit exact files with message: `feat: add staff OMR v2 task model`

---

### Task 4: Worker-Invariant Data Pipeline And Evaluation

**Files:**
- Create: `experiments/staff_level_omr/protocol/data_pipeline.py`
- Create: `experiments/staff_level_omr/protocol/evaluation.py`
- Modify: `experiments/staff_level_omr/protocol/data_bundle.py`
- Create: `tests/test_staff_level_omr_data_pipeline.py`
- Create: `tests/test_staff_level_omr_evaluation.py`

**Interfaces:**
- Produces: `EpochSampleSampler`, `StaffOMRDataset`, `build_data_loaders(...)`, `train_epoch(...)`, and `evaluate_split(...)`.
- Extends: `load_dataset_bundle(..., bundle_filename: str = "bundle.json")` so run-local `dataset_bundle.json` is loadable without duplicating data.
- Consumes: `ctc_collate`, sample-level augmentation seeds, geometry processor, CTC feasibility, and layered metrics.

- [ ] **Step 1: Write failing worker-invariance and loss tests**

```python
@pytest.mark.slow
@pytest.mark.parametrize("workers", [0, 2, 4])
def test_epoch_bytes_are_worker_invariant(workers, tiny_bundle):
    observed = collect_epoch_bytes(tiny_bundle, workers=workers)
    assert observed == collect_epoch_bytes(tiny_bundle, workers=0)

def test_boundary_ctc_sample_has_finite_loss_and_nonzero_head_gradient():
    target = torch.tensor([1, 1, 2])
    assert minimum_ctc_frames(target) == PATCH_COLS
    loss.backward()
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in model.classifier.parameters()
    )
```

- [ ] **Step 2: Run the focused tests and confirm failures**

Run: `.venv\Scripts\python.exe -m pytest tests/test_staff_level_omr_data_pipeline.py tests/test_staff_level_omr_evaluation.py -q`

- [ ] **Step 3: Implement epoch-index sampling and loaders**

Sort each epoch by `(sample_order_digest, sample_id UTF-8 bytes)` and yield `(epoch, manifest_index)`. The dataset sets its Compose instance to the full sample-level 256-bit seed immediately before augmentation. Require `in_order` in `inspect.signature(DataLoader)`, use `shuffle=False`, `drop_last=False`, `persistent_workers=False`, `pin_memory=False`, and a separately seeded loader generator. Pinning remains in the measured performance phase instead of being smuggled into the trusted baseline.

- [ ] **Step 4: Implement finite CTC training and layered evaluation**

Use concatenated variable-length targets and CPU input lengths. Raise `ProtocolError` for any non-finite loss. Evaluation always reports all-sample CER, feasible-only CER, infeasible count/rate and target-length distributions; it also reports `val_loss_feasible` without using it for checkpoint selection.

- [ ] **Step 5: Run focused tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_staff_level_omr_data_pipeline.py tests/test_staff_level_omr_evaluation.py -q`

- [ ] **Step 6: Commit the data pipeline**

Commit exact files with message: `feat: add worker invariant staff OMR pipeline`

---

### Task 5: Run Artifacts And Checkpoint Transactions

**Files:**
- Create: `experiments/staff_level_omr/protocol/artifacts.py`
- Create: `experiments/staff_level_omr/protocol/checkpoint.py`
- Create: `tests/test_staff_level_omr_artifacts.py`
- Create: `tests/test_staff_level_omr_checkpoint.py`

**Interfaces:**
- Produces: `RunArtifacts.create(...)`, atomic JSON and torch writers, fsynced `append_epoch_metrics`, `repair_metrics_jsonl`, `write_summary`, and run-sidecar transitions.
- Produces: `CHECKPOINT_SCHEMA`, `build_checkpoint(...)`, `load_checkpoint(...)`, `validate_resume_checkpoint(...)`, and `restore_trainable_state(...)`.
- Consumes: canonical hashes, vocabulary documents, ordered optimizer parameter names, and run-relative bundle paths.

- [ ] **Step 1: Write failing artifact and checkpoint tests**

```python
def test_checkpoint_embeds_vocab_but_not_manifest(checkpoint):
    assert checkpoint["vocabulary"]["tokens"]
    assert checkpoint["split_manifest_sha256"]
    assert "split_manifest" not in checkpoint

def test_metrics_repair_rejects_records_ahead_of_last_checkpoint(run):
    append_raw(run.metrics_path, valid_future_epoch_line)
    with pytest.raises(ProtocolError, match="ahead"):
        repair_metrics_jsonl(run.metrics_path, committed_epoch=2)
```

- [ ] **Step 2: Run tests and confirm missing-module failures**

Run: `.venv\Scripts\python.exe -m pytest tests/test_staff_level_omr_artifacts.py tests/test_staff_level_omr_checkpoint.py -q`

- [ ] **Step 3: Implement run-directory ownership**

Create `<UTC timestamp>-<training_contract[:12]>-<uuid[:12]>` with `exist_ok=False`. Canonically copy the four bundle documents into the run root, then use those copies as the authority. Write `run.json` status changes atomically. Append metrics as one canonical JSON line followed by flush and `fsync`.

- [ ] **Step 4: Implement exact checkpoint schema and strict loader**

Persist schema/protocol/role, identity hashes, epoch and early-stopping state, ordered optimizer names, trainable-only model state, optimizer state, embedded vocabulary, config/contracts, package versions, resume history, and artifact-relative paths. Save via a same-directory temporary file, flush/fsync it, then `os.replace`. Reject unknown schemas, wrong roles, cross-run paths, legacy state-dict-only files, identity mismatches, and optimizer-name mismatches before mutating model or optimizer.

- [ ] **Step 5: Run focused tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_staff_level_omr_artifacts.py tests/test_staff_level_omr_checkpoint.py -q`

- [ ] **Step 6: Commit artifact transactions**

Commit exact files with message: `feat: add transactional staff OMR runs`

---

### Task 6: Training, Resume, And Finalization Lifecycle

**Files:**
- Create: `experiments/staff_level_omr/protocol/contracts.py`
- Create: `experiments/staff_level_omr/protocol/runtime.py`
- Modify: `experiments/staff_level_omr/protocol/config.py`
- Create: `tests/test_staff_level_omr_runtime_integration.py`
- Create: `tests/test_staff_level_omr_resume.py`

**Interfaces:**
- Produces: `RuntimeDependencies`, `train(config, dependencies=None) -> Path`, and `resume(run_dir, overrides, dependencies=None) -> Path`.
- Produces: canonical `input_contract`, `augmentation_contract`, `task_head_contract`, `training_contract`, `launch_config`, and package-version records.
- Consumes: all protocol modules from Tasks 1-5.

- [ ] **Step 1: Write failing end-to-end CPU tests**

```python
def test_cpu_protocol_lifecycle_and_epoch_boundary_resume(tiny_bundle, deps):
    run_dir = train(config(max_epochs=1), dependencies=deps)
    assert (run_dir / "checkpoints" / "last.pt").is_file()
    assert (run_dir / "checkpoints" / "best.pt").is_file()
    assert (run_dir / "test.json").is_file()
    assert (run_dir / "summary.json").is_file()

    resume(run_dir, {"max_epochs": 2}, dependencies=deps)
    assert load_last(run_dir)["epoch"] == 2
    assert read_epochs(run_dir / "metrics.jsonl") == [1, 2]
```

Also compare the epoch-2 state and metrics against an uninterrupted two-epoch run, fail preflight before the backbone provider is called, and exercise idempotent finalization plus repair after each simulated epoch-transaction crash point.

- [ ] **Step 2: Run integration tests and confirm failures**

Run: `.venv\Scripts\python.exe -m pytest tests/test_staff_level_omr_runtime_integration.py tests/test_staff_level_omr_resume.py -q`

- [ ] **Step 3: Implement canonical contracts and initial lifecycle**

Validate config and registry metadata without downloading weights, load/verify the bundle, derive all contracts and three identity hashes, create the run, write canonical copies and `run.json`, run augmentation and CTC preflight, and mark failed preflight runs with candidate exclusions before re-raising. Only after successful preflight reset the init seed and invoke the backbone provider.

- [ ] **Step 4: Implement epoch transactions and stopping**

For each epoch reset main-process RNG, build the epoch loader, train, evaluate from `start_eval`, update best/patience state, atomically write `last.pt`, write `best.pt` only on improvement, append metrics, and update `run.json`. Early stopping has priority over max epochs. Non-finite values fail the run without claiming completion.

- [ ] **Step 5: Implement strict epoch-boundary resume**

Reconstruct identity from run-local documents and checkpoint config. Hard-reject protocol, vocabulary, manifest, model revision, model weights, geometry, method, seed, augmentation, task-head, optimizer, and major/minor core-package drift. Permit paths and workers to change. Permit only an increased `max_epochs` and require explicit `allow_env_drift` for soft patch drift. Recompute epoch RNG instead of serializing RNG objects, repair sidecars from `last.pt`, and record every accepted override.

- [ ] **Step 6: Implement idempotent finalization**

Load `best.pt`, evaluate test once for the matching identity, write `test.json`, calculate artifact hashes into `summary.json`, and mark the run complete. A terminal resume only repeats missing or invalid finalization unless `max_epochs` is legally extended.

- [ ] **Step 7: Run lifecycle tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_staff_level_omr_runtime_integration.py tests/test_staff_level_omr_resume.py -q`

- [ ] **Step 8: Commit the runtime**

Commit exact files with message: `feat: implement staff OMR v2 lifecycle`

---

### Task 7: Atomic Public CLI Replacement

**Files:**
- Modify: `musvit/cli.py`
- Replace: `experiments/staff_level_omr/arguments.py`
- Replace: `experiments/staff_level_omr/entrypoint.py`
- Replace: `experiments/staff_level_omr/train.py`
- Replace: `experiments/staff_level_omr/model.py`
- Replace: `experiments/staff_level_omr/datasets.py`
- Replace: `experiments/staff_level_omr/augments.py`
- Delete: `experiments/staff_level_omr/config.py`
- Delete: `experiments/staff_level_omr/utils/data_utils.py`
- Delete: `experiments/staff_level_omr/utils/utils.py`
- Modify: `tests/test_staff_level_omr_prepare_data_cli.py`
- Modify: `tests/test_staff_level_omr_launcher.py`
- Create: `tests/test_staff_level_omr_train_cli.py`

**Interfaces:**
- Public commands: `musvit staff-level-omr prepare-data`, `musvit staff-level-omr train`, and `musvit staff-level-omr resume`.
- Direct module: `python -m experiments.staff_level_omr.train`.
- Compatibility: omitted `train`, `linear_prob`, and `shape_patches` normalize into the same `StaffOMRConfig`.

- [ ] **Step 1: Write failing CLI normalization and rejection tests**

```python
def test_explicit_and_omitted_train_select_same_callable():
    assert select(["staff-level-omr", "--experiment_name=x"]) == select(
        ["staff-level-omr", "train", "--experiment_name=x"]
    )

def test_old_and_new_patch_arguments_together_are_rejected(tmp_path):
    with pytest.raises(ProtocolError, match="cannot be combined"):
        make_config(patch_rows=8, patch_cols=64, shape_patches=(8, 64))
```

- [ ] **Step 2: Run CLI tests and confirm legacy behavior fails them**

Run: `.venv\Scripts\python.exe -m pytest tests/test_staff_level_omr_prepare_data_cli.py tests/test_staff_level_omr_train_cli.py tests/test_staff_level_omr_launcher.py -q`

- [ ] **Step 3: Replace public adapters and remove executable v1 code**

Make every adapter construct `StaffOMRConfig` from the pinned registry and call `protocol.runtime.train` or `protocol.runtime.resume`. The root CLI exposes the three subcommands while preserving omitted-train routing. Delete legacy split, LabelEncoder, global target padding, hard-coded-worker, 1000-epoch loop, state-dict-only checkpoint, and `zero_infinity=True` implementations rather than hiding them behind a v1 flag.

- [ ] **Step 4: Run CLI and import tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_staff_level_omr_prepare_data_cli.py tests/test_staff_level_omr_train_cli.py tests/test_staff_level_omr_launcher.py -q`

- [ ] **Step 5: Commit the atomic switch**

Commit exact staff runtime and CLI files with message: `refactor: switch staff OMR CLI to protocol v2`

---

### Task 8: Documentation And Complete Acceptance

**Files:**
- Replace: `experiments/staff_level_omr/README.md`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-07-25-staff-level-omr-trusted-training-protocol-v2-design.md` only if implementation discoveries require a normative clarification.
- Modify: the runtime/tests above for acceptance defects found during verification.

**Interfaces:**
- Documents only the executable v2 commands, artifacts, failure modes, resume matrix, aliases, and the one-time linear-probe geometry comparison requirement.

- [ ] **Step 1: Rewrite README against actual CLI help**

Document `prepare-data`, `train`, and `resume` with required paths; explain full versus feasible CER, candidate exclusions, run-local identities, no normalization after `ToTensor`, pinned model evidence, and the expected `native_pad` linear-probe compatibility risk. Remove claims that training evaluates train CER, prints prediction pairs, or that invoking `augments.py` writes sample images.

- [ ] **Step 2: Run every staff-level test with local Cairo available**

Run:

```powershell
$env:PATH='D:\BaiduNetdisk\module\ImageViewer;' + $env:PATH
.venv\Scripts\python.exe -m pytest tests/test_staff_level_omr*.py -q
```

Expected: all staff-level tests pass.

- [ ] **Step 3: Run the complete repository suite**

Run:

```powershell
$env:PATH='D:\BaiduNetdisk\module\ImageViewer;' + $env:PATH
.venv\Scripts\python.exe -m pytest -q
```

Expected: every staff-level test passes; any remaining failure must be proven unrelated by reproducing it against the pre-task tree or isolating it to the already-dirty full-page files.

- [ ] **Step 4: Audit section 18 mechanically**

Search for forbidden legacy behavior and hard-coded geometry:

```powershell
rg -n "zero_infinity=True|LabelEncoder|train_test_split|reshape\\([^\\n]*64[^\\n]*64|1024\\s*-" experiments/staff_level_omr
```

Expected: no executable v2 hit. Inspect every section-18 acceptance item and name the passing test that proves it.

- [ ] **Step 5: Run compile and clean-diff checks**

Run:

```powershell
.venv\Scripts\python.exe -m compileall -q experiments/staff_level_omr musvit
git diff --check
git status --short
```

Expected: compilation and whitespace checks succeed; only task files plus preserved unrelated user changes remain.

- [ ] **Step 6: Commit documentation and final corrections**

Commit exact task files with message: `docs: complete staff OMR v2 protocol`

- [ ] **Step 7: Apply the finishing workflow**

Use `superpowers:verification-before-completion`, then `superpowers:finishing-a-development-branch`. Do not claim completion until the acceptance matrix and test outputs are current.
