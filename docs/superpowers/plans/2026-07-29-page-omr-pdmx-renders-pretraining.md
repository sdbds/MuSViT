# page-omr-pdmx-renders Native Pretraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible, local-only WebDataset pretraining path for `page-omr-pdmx-renders`, with an append-only Polish-based vocabulary, strict dataset manifests, honest no-test semantics, and token-aware checkpoint migration.

**Architecture:** Keep the existing Arrow and curriculum code unchanged. Add focused modules for deterministic vocabulary artifacts, offline tar scanning, and a PDMX `IterableDataset`/Lightning DataModule; integrate them through a tagged configuration and a new `PDMX` training regime. Use virtual epochs with deterministic single-GPU streams and explicitly support only virtual-epoch-boundary full resume in v1.

**Tech Stack:** Python 3.11, PyTorch 2.13, Lightning 2.6, Hugging Face Hub, WebDataset, Pillow, NumPy, pytest.

## Global Constraints

- Dataset id is `tobiashornbogen/page-omr-pdmx-renders`.
- Dataset revision is exactly `7da3ae5237963e57a8fe1c6ee375b1f10af34a09`.
- Training is local-only and must never download missing shards implicitly.
- Batch size is exactly 1, input resolution is 1024, model maxlen is 7512, and tokenization is `bekern`.
- Existing Arrow configs without `data.type` remain valid and keep their current loader behavior.
- Existing Polish, Mozarteum, FP GrandStaff vocabulary files and checkpoints are not rewritten.
- `FullPageOMR_BeKern_v1` preserves Polish ids 0 through 214, appends the six current compatibility tokens at ids 215 through 220, then appends UTF-8-byte-sorted PDMX train-only tokens.
- Validation cannot expand the vocabulary; all validation OOVs are fatal.
- Runtime vocabulary growth, `<unk>`, silent sequence truncation, warning-and-skip corruption handling, and fake test splits are prohibited.
- PDMX v1 uses both renderers at weights `verovio=0.5`, `mscore=0.5`, with runtime augmentation disabled.
- PDMX v1 is single-process, single-GPU. Full resume is accepted only at virtual epoch boundaries with identical stream topology.
- Every task follows red-green-refactor TDD and commits only its own files.

---

## File Responsibility Map

- `experiments/full_page_omr/utils/vocab_manifest.py`: canonical vocabulary representation, digesting, legacy `.npy` validation/conversion, append-only extension, and JSON persistence.
- `experiments/full_page_omr/pdmx_manifest.py`: strict tar pairing/scanning, local Hugging Face snapshot resolution, shard hashing, dataset manifest construction, and CLI.
- `experiments/full_page_omr/pdmx_data.py`: runtime WebDataset streams, deterministic renderer/shard/sample selection, tensor conversion, validation materialization, and Lightning DataModule.
- `experiments/full_page_omr/config/ExperimentConfigWrapper.py`: backward-compatible tagged union between legacy Arrow data and PDMX data.
- `experiments/full_page_omr/finetune.py`: register the new regime, attach virtual-epoch callback, include data/vocabulary identity in protocol metadata, and skip test only when the DataModule explicitly has no test split.
- `experiments/full_page_omr/smt_trainer.py`: save and validate stream/vocabulary checkpoint metadata without changing model architecture.
- `experiments/full_page_omr/migrate_vocabulary_checkpoint.py`: pure token-axis remapping and CLI/reporting for weights-only migration.
- `experiments/full_page_omr/config/Page_OMR_PDMX/pretraining.json`: fixed PDMX data protocol.
- `experiments/full_page_omr/config/Polish_Scores/pdmx_finetuning.json`: Polish downstream config that retains the unified vocabulary.
- `experiments/full_page_omr/config/Mozarteum/pdmx_finetuning.json`: Mozarteum downstream config that retains the unified vocabulary.
- `tests/test_full_page_omr_vocab_manifest.py`: deterministic vocabulary and legacy conversion tests.
- `tests/test_full_page_omr_pdmx_manifest.py`: tar scanner and dataset manifest tests.
- `tests/test_full_page_omr_pdmx_data.py`: stream, renderer mix, image, virtual epoch, and no-test tests.
- `tests/test_full_page_omr_vocab_migration.py`: checkpoint token-axis remapping tests.

---

### Task 1: Deterministic Vocabulary Manifest

**Files:**
- Create: `experiments/full_page_omr/utils/vocab_manifest.py`
- Create: `tests/test_full_page_omr_vocab_manifest.py`

**Interfaces:**
- Produces: `VocabularyManifest`, `canonical_json_sha256()`, `load_legacy_ordered_tokens()`, `build_project_seed_tokens()`, `extend_ordered_tokens()`, `load_vocabulary_manifest()`, `write_vocabulary_manifest()`, and `write_legacy_numpy_pair()`.
- Consumes: the six existing `.npy` vocabulary artifacts under `experiments/full_page_omr/vocab/`.

- [ ] **Step 1: Write failing tests for the Polish prefix and legacy bijection**

```python
def test_project_seed_preserves_polish_ids_and_appends_known_extras():
    seed = build_project_seed_tokens(VOCAB_DIR)
    assert len(seed) == 221
    assert ordered_token_sha256(seed[:215]) == POLISH_PREFIX_SHA256
    assert seed[215:] == (
        "*M6/16", "*staff1", "*staff2", "88", "=:|!;", "==;"
    )
    assert seed[0] == "<pad>"
    assert seed[44] == "<s>"
    assert seed[100] == "<bos>"
    assert seed[132] == "<b>"
    assert seed[183] == "<eos>"
    assert seed[29] == "<t>"


def test_legacy_loader_rejects_non_inverse_maps(tmp_path):
    write_npy_pair(tmp_path, {"<pad>": 0, "x": 1}, {0: "<pad>", 1: "y"})
    with pytest.raises(ValueError, match="not strict inverses"):
        load_legacy_ordered_tokens(tmp_path / "w2i.npy", tmp_path / "i2w.npy")
```

- [ ] **Step 2: Run the focused tests and verify red**

Run:

```powershell
uv run pytest tests/test_full_page_omr_vocab_manifest.py -q
```

Expected: collection fails because `vocab_manifest` does not exist.

- [ ] **Step 3: Implement canonical digests and strict legacy loading**

Implement these exact signatures:

```python
POLISH_PREFIX_SHA256 = "3821e7f0d5defd55fe73ce8b7f229f48f688bc490bdae854bc824a6145dac49b"

def canonical_json_sha256(value: Any) -> str: ...
def ordered_token_sha256(tokens: Sequence[str]) -> str: ...
def load_legacy_ordered_tokens(
    w2i_path: str | Path,
    i2w_path: str | Path,
) -> tuple[str, ...]: ...
def build_project_seed_tokens(vocab_dir: str | Path) -> tuple[str, ...]: ...
```

`load_legacy_ordered_tokens()` must reject booleans as ids, gaps, duplicates, non-string tokens, missing files, and maps that are not exact inverses. `build_project_seed_tokens()` must load Polish first, verify its 215-token digest, compute the Mozarteum/FP union difference, verify it is exactly the approved six-token set, and append it by UTF-8 byte order.

- [ ] **Step 4: Write failing tests for deterministic append-only extension**

```python
def test_extension_is_order_independent_and_append_only():
    base = ("<pad>", "z")
    first = extend_ordered_tokens(base, [["é", "a"], ["ß", "a"]])
    second = extend_ordered_tokens(base, [["ß"], ["a", "é"]])
    assert first == second
    assert first[:2] == base
    assert first[2:] == tuple(sorted({"é", "a", "ß"}, key=lambda t: t.encode("utf-8")))
```

- [ ] **Step 5: Implement the immutable JSON vocabulary artifact**

```python
@dataclass(frozen=True)
class VocabularyManifest:
    schema_version: int
    name: str
    tokenization_mode: str
    base_name: str
    base_size: int
    base_digest: str
    ordered_tokens: tuple[str, ...]
    token_provenance: dict[str, str]
    source_dataset_manifests: tuple[str, ...]
    vocab_sha256: str

    @property
    def w2i(self) -> dict[str, int]: ...

    @property
    def i2w(self) -> dict[int, str]: ...


def extend_ordered_tokens(
    base_tokens: Sequence[str],
    token_sequences: Iterable[Iterable[str]],
) -> tuple[str, ...]: ...
def load_vocabulary_manifest(path: str | Path) -> VocabularyManifest: ...
def write_vocabulary_manifest(
    manifest: VocabularyManifest,
    path: str | Path,
) -> Path: ...
def write_legacy_numpy_pair(
    manifest: VocabularyManifest,
    output_dir: str | Path,
) -> tuple[Path, Path]: ...
```

Loading must recompute and verify `vocab_sha256`; writing must use UTF-8, sorted object keys, compact separators, and a trailing newline. Do not call the legacy `make_vocabulary()`.
The compatibility writer derives both maps only from `ordered_tokens`, writes
`<name>w2i.npy/<name>i2w.npy`, reloads them through `load_legacy_ordered_tokens()`, and fails unless the
round trip exactly matches the JSON order.

- [ ] **Step 6: Run focused and legacy vocabulary tests**

Run:

```powershell
uv run pytest tests/test_full_page_omr_vocab_manifest.py tests/test_full_page_omr_vocab.py -q
```

Expected: all pass.

- [ ] **Step 7: Commit Task 1**

```powershell
git add experiments/full_page_omr/utils/vocab_manifest.py tests/test_full_page_omr_vocab_manifest.py
git commit -m "feat: add deterministic OMR vocabulary manifests"
```

---

### Task 2: Strict PDMX Tar and Dataset Manifest Scanner

**Files:**
- Create: `experiments/full_page_omr/pdmx_manifest.py`
- Create: `tests/test_full_page_omr_pdmx_manifest.py`

**Interfaces:**
- Consumes: `parse_kern_file()` and Task 1 digest/extension functions.
- Produces: `PDMXSampleRecord`, `ShardScanResult`, `scan_pdmx_tar()`, `build_pdmx_dataset_manifest()`, `verify_pdmx_dataset_manifest()`, and CLI commands `prepare` and `scan`.

- [ ] **Step 1: Write a reusable tiny-tar fixture and failing strict-pairing tests**

```python
def test_scan_requires_image_kern_and_source(tmp_path):
    tar_path = write_pdmx_tar(
        tmp_path / "broken.tar",
        {"a": {"image.png": png_bytes(), "kern.txt": VALID_KERN.encode()}},
    )
    with pytest.raises(ValueError, match=r"a.*source\.txt"):
        scan_pdmx_tar(tar_path, renderer="verovio", logical_path="broken.tar")


def test_scan_accepts_optional_fill_and_records_token_lengths(tmp_path):
    tar_path = write_pdmx_tar(
        tmp_path / "valid.tar",
        {
            "a": {
                "image.png": png_bytes((48, 32)),
                "kern.txt": VALID_KERN.encode(),
                "source.txt": b"scores/a.mxl\n",
                "fill.txt": b"medium\n",
            }
        },
    )
    result = scan_pdmx_tar(
        tar_path,
        renderer="verovio",
        logical_path="train/valid.tar",
    )
    assert result.sample_count == 1
    assert result.records[0].source_id == "scores/a.mxl"
    assert result.records[0].sequence_length >= 2
```

- [ ] **Step 2: Run the scanner tests and verify red**

Run:

```powershell
uv run pytest tests/test_full_page_omr_pdmx_manifest.py -q
```

Expected: import fails because `pdmx_manifest` does not exist.

- [ ] **Step 3: Implement strict tar grouping and shard scanning**

```python
@dataclass(frozen=True)
class PDMXSampleRecord:
    key: str
    source_id: str
    fill: str | None
    sequence_length: int
    tokens: tuple[str, ...]
    image_size: tuple[int, int]


@dataclass(frozen=True)
class ShardScanResult:
    renderer: str
    logical_path: str
    sha256: str
    bytes: int
    sample_count: int
    records: tuple[PDMXSampleRecord, ...]


def scan_pdmx_tar(
    tar_path: str | Path,
    *,
    renderer: str,
    logical_path: str,
    max_sequence_length: int = 7512,
) -> ShardScanResult: ...
```

Recognize suffixes from the right so keys may contain dots. Decode Kern/source strictly as UTF-8, verify PNG with Pillow, call the shared BeKern parser, add BOS/EOS before length validation, normalize source with `.strip()`, and fail with logical path plus key for every malformed record.

- [ ] **Step 4: Write failing tests for manifest identity and source overlap**

```python
def test_manifest_rejects_train_validation_source_overlap(tmp_path):
    snapshot = make_snapshot(tmp_path, train_source="same.mxl", val_source="same.mxl")
    with pytest.raises(ValueError, match="train/validation source overlap"):
        build_pdmx_dataset_manifest(
            snapshot_root=snapshot,
            dataset_id=DATASET_ID,
            dataset_revision=DATASET_REVISION,
            train_shards=TRAIN_SELECTION,
            validation_shards=VAL_SELECTION,
            renderer_weights={"verovio": 0.5, "mscore": 0.5},
        )


def test_manifest_digest_is_independent_of_input_shard_order(tmp_path):
    first = build_manifest_for_fixture(tmp_path, reverse=False)
    second = build_manifest_for_fixture(tmp_path, reverse=True)
    assert first["manifest_sha256"] == second["manifest_sha256"]
```

- [ ] **Step 5: Implement dataset manifest build, load, and local verification**

```python
def build_pdmx_dataset_manifest(
    *,
    snapshot_root: str | Path,
    dataset_id: str,
    dataset_revision: str,
    train_shards: Sequence[tuple[str, str]],
    validation_shards: Sequence[str],
    renderer_weights: Mapping[str, float],
    max_sequence_length: int = 7512,
) -> dict[str, Any]: ...

def write_pdmx_dataset_manifest(manifest: Mapping[str, Any], path: str | Path) -> Path: ...
def load_pdmx_dataset_manifest(path: str | Path) -> dict[str, Any]: ...
def verify_pdmx_dataset_manifest(
    manifest: Mapping[str, Any],
    snapshot_root: str | Path,
) -> None: ...
```

The manifest must sort logical paths, record hashes/counts/token frequencies/length summaries/exclusions, require exactly the two approved renderer weights, and reject source overlap. `verify_pdmx_dataset_manifest()` rechecks file existence, size, SHA-256, id, revision, and self-digest before model construction.

- [ ] **Step 6: Add local-only Hugging Face snapshot resolution and CLI**

```python
def resolve_local_snapshot(
    dataset_id: str,
    revision: str,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    return Path(snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=True,
    ))
```

The separate `prepare` CLI may call `snapshot_download(..., local_files_only=False, allow_patterns=...)`; `scan` must resolve local-only. A missing snapshot must raise a message containing the exact prepare command and never retry against `main`.

- [ ] **Step 7: Run scanner tests**

Run:

```powershell
uv run pytest tests/test_full_page_omr_pdmx_manifest.py -q
```

Expected: all pass without network.

- [ ] **Step 8: Commit Task 2**

```powershell
git add experiments/full_page_omr/pdmx_manifest.py tests/test_full_page_omr_pdmx_manifest.py
git commit -m "feat: add strict PDMX dataset manifests"
```

---

### Task 3: Backward-Compatible Tagged Data Configuration

**Files:**
- Modify: `experiments/full_page_omr/config/ExperimentConfigWrapper.py`
- Modify: `tests/test_full_page_omr_config.py`

**Interfaces:**
- Produces: `PDMXData`, `DataConfig = Data | PDMXData`, and compatible `ExperimentConfig`.
- Consumes: existing `Data` and `ExperimentConfig` JSON behavior.

- [ ] **Step 1: Add failing tests for old-config compatibility and new strict parsing**

```python
def test_existing_arrow_config_defaults_to_legacy_data_type():
    config = experiment_config_from_dict(POLISH_CONFIG)
    assert isinstance(config.data, Data)
    assert config.to_dict() == POLISH_CONFIG


def test_pdmx_config_parses_and_exposes_legacy_neutral_properties():
    config = experiment_config_from_dict(PDMX_CONFIG)
    assert isinstance(config.data, PDMXData)
    assert config.data.skip_steps == 0
    assert config.data.reduce_ratio == 1.0
    assert config.data.renderer_weights == {"verovio": 0.5, "mscore": 0.5}


@pytest.mark.parametrize("field", ["reduce_ratio", "skip_steps"])
def test_pdmx_config_rejects_arrow_only_fields(field):
    payload = copy.deepcopy(PDMX_CONFIG)
    payload["data"][field] = 1
    with pytest.raises(ValueError, match=field):
        experiment_config_from_dict(payload)
```

- [ ] **Step 2: Run the config tests and verify red**

Run:

```powershell
uv run pytest tests/test_full_page_omr_config.py -q
```

Expected: PDMX cases fail because `PDMXData` does not exist; existing cases stay green.

- [ ] **Step 3: Implement `PDMXData` and tagged parsing**

```python
@dataclass
class PDMXData:
    dataset_id: str
    dataset_revision: str
    dataset_manifest: str
    vocab_manifest: str
    renderer_weights: dict[str, float]
    batch_size: int
    num_workers: int
    tokenization_mode: str
    steps_per_epoch: int
    shuffle_buffer: int
    seed: int
    runtime_augmentation: bool

    @property
    def skip_steps(self) -> int:
        return 0

    @property
    def reduce_ratio(self) -> float:
        return 1.0
```

`ExperimentConfig.from_dict()` dispatches on `data.type`; a missing type calls the current `Data.from_dict()` unchanged. PDMX parsing validates the exact 40-character revision, approved id, exact renderer keys/weights, positive lengths, batch size 1, BeKern, and false augmentation. Reject unknown PDMX keys. `to_dict()` must round-trip both variants.

- [ ] **Step 4: Run config and entrypoint regression tests**

Run:

```powershell
uv run pytest tests/test_full_page_omr_config.py tests/test_full_page_omr_throughput.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit Task 3**

```powershell
git add experiments/full_page_omr/config/ExperimentConfigWrapper.py tests/test_full_page_omr_config.py
git commit -m "feat: add PDMX data configuration"
```

---

### Task 4: Native WebDataset Runtime and Lightning DataModule

**Files:**
- Create: `experiments/full_page_omr/pdmx_data.py`
- Create: `tests/test_full_page_omr_pdmx_data.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: Task 1 vocabulary manifest, Task 2 dataset manifest, Task 3 `PDMXData`, shared `parse_kern_file()`, `convert_img_to_tensor()`, and `batch_preparation_img2seq()`.
- Produces: `PDMXTrainDataset`, `PDMXValidationDataset`, `PDMXPretrainingDataModule`, `PDMXVirtualEpochCallback`, and `PDMXConsumptionAuditCallback`.

- [ ] **Step 1: Add WebDataset as a locked dependency**

Run:

```powershell
uv add webdataset
```

Expected: `pyproject.toml` and `uv.lock` contain the resolved package; no unrelated dependency is removed.

- [ ] **Step 2: Write failing tests for train/validation samples**

```python
def test_train_dataset_yields_model_contract_and_metadata(pdmx_fixture):
    dataset = PDMXTrainDataset.from_fixture(
        pdmx_fixture,
        steps_per_epoch=8,
        num_workers_contract=0,
    )
    sample = next(iter(dataset))
    image, decoder_input, target, metadata = sample
    assert image.shape == (1, 3, 1024, 1024)
    assert decoder_input.shape == target.shape
    assert metadata["renderer"] in {"verovio", "mscore"}
    assert metadata["sample_key"]


def test_validation_dataset_is_ordered_and_variable_size_safe(pdmx_fixture):
    dataset = PDMXValidationDataset.from_fixture(pdmx_fixture)
    assert [dataset[i][3]["sample_key"] for i in range(len(dataset))] == ["val-a", "val-b"]
    assert all(dataset[i][0].shape == (1, 3, 1024, 1024) for i in range(len(dataset)))
```

- [ ] **Step 3: Run the focused data tests and verify red**

Run:

```powershell
uv run pytest tests/test_full_page_omr_pdmx_data.py -q
```

Expected: import fails because `pdmx_data` does not exist.

- [ ] **Step 4: Implement strict WebDataset record decoding**

Use `webdataset.WebDataset` on explicit local shard paths with library-level shard and worker splitting disabled because the dataset owns both. Require record keys `image.png`, `kern.txt`, and `source.txt`; decode images as RGB; tokenize with the shared parser; map every token through the frozen `w2i`; and raise contextual errors for runtime drift.

Implement sample-scoped teacher forcing:

```python
def deterministic_teacher_forcing(
    target: torch.Tensor,
    *,
    vocab_size: int,
    padding_token: int,
    probability: float,
    seed_material: str,
) -> torch.Tensor: ...
```

Hash `seed_material` with SHA-256 and seed a local NumPy generator. Do not mutate global RNG state.

- [ ] **Step 5: Write failing virtual-epoch and renderer tests**

```python
@pytest.mark.parametrize("num_workers", [0, 2])
def test_virtual_epoch_has_exact_global_length(pdmx_fixture, num_workers):
    loader = build_fixture_loader(pdmx_fixture, steps_per_epoch=17, num_workers=num_workers)
    rows = list(loader)
    assert len(rows) == 17


def test_renderer_sequence_is_reproducible_for_fixed_topology(pdmx_fixture):
    first = collect_metadata(pdmx_fixture, seed=3407, epoch=3, num_workers=0)
    second = collect_metadata(pdmx_fixture, seed=3407, epoch=3, num_workers=0)
    assert first == second
    assert {row["renderer"] for row in first} == {"verovio", "mscore"}
```

- [ ] **Step 6: Implement deterministic renderer streams**

```python
class PDMXTrainDataset(torch.utils.data.IterableDataset):
    def set_virtual_epoch(self, epoch: int) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self): ...


class PDMXValidationDataset(torch.utils.data.Dataset):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int): ...
```

Assign global ordinals to workers by stride so worker quotas sum to exactly `steps_per_epoch`. Derive renderer decisions, shard orders, bounded sample shuffles, source cycles, and teacher-forcing seeds from separate SHA-256 namespaces. Within one renderer/worker cycle, iterate finite shards without replacement before incrementing `source_cycle`.

- [ ] **Step 7: Implement the Lightning DataModule and epoch callback**

```python
class PDMXPretrainingDataModule(LightningDataModule):
    has_validation_split = True
    has_test_split = False
    encoder_unfreeze_step = 0
    curriculum_step_offset = 0
    stream_resume_mode = "virtual_epoch_boundary"

    def set_train_epoch(self, epoch: int) -> None: ...
    def protocol_metadata(self) -> dict[str, Any]: ...
    def train_dataloader(self): ...
    def val_dataloader(self): ...


class PDMXVirtualEpochCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module) -> None:
        trainer.datamodule.set_train_epoch(trainer.current_epoch)


class PDMXConsumptionAuditCallback(Callback):
    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ) -> None: ...
    def on_train_epoch_end(self, trainer, pl_module) -> None: ...
```

Use `prefetch_factor=1`, ordered delivery, `persistent_workers=False`, no DataLoader shuffle, and existing batch-size-1 collate. Do not implement `test_dataloader()`.
The audit callback consumes only metadata returned to the main process. It records actual renderer/source-cycle/
voice-bucket/density-bucket counts per virtual epoch, logs renderer ratios, and saves the first processed image
plus original/final dimensions for each renderer under the run-instance directory. It raises when a completed
virtual epoch contains no sample from either enabled renderer.

- [ ] **Step 8: Run data, worker, and legacy pipeline tests**

Run:

```powershell
uv run pytest tests/test_full_page_omr_pdmx_data.py tests/test_full_page_omr_data_pipeline.py -q
```

Expected: all pass.

- [ ] **Step 9: Commit Task 4**

```powershell
git add pyproject.toml uv.lock experiments/full_page_omr/pdmx_data.py tests/test_full_page_omr_pdmx_data.py
git commit -m "feat: add PDMX WebDataset data module"
```

---

### Task 5: Training Regime, Protocol Metadata, and Honest No-Test Semantics

**Files:**
- Modify: `experiments/full_page_omr/finetune.py`
- Modify: `experiments/full_page_omr/smt_trainer.py`
- Modify: `tests/test_full_page_omr_throughput.py`
- Modify: `tests/test_full_page_omr_model_contracts.py`

**Interfaces:**
- Consumes: Task 4 DataModule/callback and `protocol_metadata()`.
- Produces: `DATASETS_TYPE["PDMX"]`, capability-based final evaluation, PDMX protocol snapshot fields, and epoch-boundary full-resume validation.

- [ ] **Step 1: Write failing tests for PDMX registration and no-test behavior**

```python
def test_pdmx_regime_is_registered():
    assert DATASETS_TYPE["PDMX"] is PDMXPretrainingDataModule


def test_final_evaluation_skips_only_explicit_no_test_datamodule():
    trainer = Mock()
    data = SimpleNamespace(has_test_split=False)
    assert _run_test_if_available(trainer, Mock(), data, "model.ckpt") is False
    trainer.test.assert_not_called()


def test_final_evaluation_runs_for_legacy_datamodule_without_capability():
    trainer = Mock()
    assert _run_test_if_available(trainer, Mock(), object(), "model.ckpt") is True
    trainer.test.assert_called_once()
```

- [ ] **Step 2: Run focused training tests and verify red**

Run:

```powershell
uv run pytest tests/test_full_page_omr_throughput.py -q
```

Expected: PDMX registration and `_run_test_if_available` tests fail.

- [ ] **Step 3: Register PDMX and make final test capability-based**

Add the new DataModule to `DATASETS_TYPE`. Add both PDMX callbacks only when the DataModule exposes their
required methods. Replace the unconditional tail call with:

```python
def _run_test_if_available(trainer, model_wrapper, data, checkpoint_path) -> bool:
    if getattr(data, "has_test_split", True) is False:
        logger.info("Skipping test: data module explicitly declares no test split")
        return False
    _run_test(trainer, model_wrapper, data, checkpoint_path)
    return True
```

Legacy modules default to true to preserve behavior. Do not catch missing-test exceptions.

- [ ] **Step 4: Write failing protocol and resume-boundary tests**

```python
def test_pdmx_protocol_contains_dataset_vocab_and_stream_identity(pdmx_data_module):
    metadata = _data_protocol_metadata(pdmx_data_module)
    assert metadata["dataset_revision"] == DATASET_REVISION
    assert metadata["dataset_manifest_sha256"]
    assert metadata["vocab_sha256"]
    assert metadata["stream_resume_mode"] == "virtual_epoch_boundary"


def test_pdmx_full_resume_rejects_mid_virtual_epoch():
    with pytest.raises(ValueError, match="virtual epoch boundary"):
        _validate_stream_resume_boundary(
            samples_seen=10_001,
            steps_per_epoch=10_000,
            mode="virtual_epoch_boundary",
        )
```

- [ ] **Step 5: Integrate data protocol metadata and callback**

Implement:

```python
def _data_protocol_metadata(data) -> dict[str, Any]:
    provider = getattr(data, "protocol_metadata", None)
    return provider() if provider is not None else {}


def _validate_stream_resume_boundary(
    *,
    samples_seen: int,
    steps_per_epoch: int,
    mode: str,
) -> None: ...
```

Merge DataModule metadata into `protocol_snapshot`, pass the exact vocabulary size/digest into model/checkpoint metadata, and add `PDMXVirtualEpochCallback` only for DataModules exposing `set_train_epoch`. On PDMX full resume, require `samples_seen % steps_per_epoch == 0`; weights-only starts remain unrestricted.

- [ ] **Step 6: Run full-page training contract tests**

Run:

```powershell
uv run pytest tests/test_full_page_omr_throughput.py tests/test_full_page_omr_model_contracts.py tests/test_full_page_omr_checkpoint_export.py -q
```

Expected: all pass.

- [ ] **Step 7: Commit Task 5**

```powershell
git add experiments/full_page_omr/finetune.py experiments/full_page_omr/smt_trainer.py tests/test_full_page_omr_throughput.py tests/test_full_page_omr_model_contracts.py
git commit -m "feat: integrate PDMX pretraining regime"
```

---

### Task 6: Token-Aware Weights-Only Checkpoint Migration

**Files:**
- Create: `experiments/full_page_omr/migrate_vocabulary_checkpoint.py`
- Create: `tests/test_full_page_omr_vocab_migration.py`
- Modify: `experiments/full_page_omr/finetune.py`

**Interfaces:**
- Consumes: Task 1 vocabulary manifests and Lightning checkpoint `state_dict`.
- Produces: `remap_vocabulary_state_dict()`, `load_vocabulary_aware_weights()`, migration report JSON, and optional `source_vocab_manifest` launch argument.

- [ ] **Step 1: Write failing pure state-dict migration tests**

```python
def test_token_axes_are_copied_by_name_not_source_id():
    source_tokens = ("<pad>", "b", "a")
    target_tokens = ("<pad>", "a", "b", "new")
    source = fake_source_state(vocab_size=3)
    target = fake_target_state(vocab_size=4)
    migrated, report = remap_vocabulary_state_dict(
        source,
        target,
        source_tokens=source_tokens,
        target_tokens=target_tokens,
    )
    assert torch.equal(migrated[EMBEDDING_KEY][1], source[EMBEDDING_KEY][2])
    assert torch.equal(migrated[OUTPUT_WEIGHT_KEY][2], source[OUTPUT_WEIGHT_KEY][1])
    assert torch.equal(migrated[OUTPUT_BIAS_KEY][1], source[OUTPUT_BIAS_KEY][2])
    assert torch.equal(migrated[EMBEDDING_KEY][3], target[EMBEDDING_KEY][3])
    assert report["new_tokens"] == ["new"]


def test_source_only_token_is_fatal():
    with pytest.raises(ValueError, match="missing from target"):
        remap_vocabulary_state_dict(
            fake_source_state(2),
            fake_target_state(1),
            source_tokens=("<pad>", "lost"),
            target_tokens=("<pad>",),
        )
```

- [ ] **Step 2: Run migration tests and verify red**

Run:

```powershell
uv run pytest tests/test_full_page_omr_vocab_migration.py -q
```

Expected: import fails because migration module does not exist.

- [ ] **Step 3: Implement strict token-axis remapping**

```python
TOKEN_AXIS_KEYS = (
    "model.decoder.embedding.weight",
    "model.decoder.out_layer.weight",
    "model.decoder.out_layer.bias",
)

def remap_vocabulary_state_dict(
    source_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    *,
    source_tokens: Sequence[str],
    target_tokens: Sequence[str],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]: ...
```

All non-token tensors must have identical key sets, shapes, and dtypes. Clone target token tensors first, copy common rows by token name, keep target initialization for new rows, and fail on source-only tokens. The report lists source/target digests, common/new/dropped tokens, and copied tensor counts.

- [ ] **Step 4: Implement checkpoint loading and report persistence**

```python
def load_vocabulary_aware_weights(
    model_wrapper: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    source_vocab_manifest: str | Path,
    target_vocab_manifest: str | Path,
    report_path: str | Path,
) -> dict[str, Any]: ...
```

Load the Lightning checkpoint with CPU mmap, read only `state_dict`, remap into the freshly initialized wrapper, call `load_state_dict(..., strict=True)`, write canonical report JSON, and never restore optimizer/scheduler/global-step/cursor state.

- [ ] **Step 5: Add an explicit launch argument without changing legacy behavior**

Add `source_vocab_manifest: str | None = None`. It is valid only with `starting_weights`; when present, construct a fresh `SMTPP_Trainer` and call the migration loader. Without it, preserve the current exact-vocabulary `load_from_checkpoint` path.

- [ ] **Step 6: Run migration and checkpoint regression tests**

Run:

```powershell
uv run pytest tests/test_full_page_omr_vocab_migration.py tests/test_full_page_omr_checkpoint_export.py tests/test_full_page_omr_throughput.py -q
```

Expected: all pass.

- [ ] **Step 7: Commit Task 6**

```powershell
git add experiments/full_page_omr/migrate_vocabulary_checkpoint.py experiments/full_page_omr/finetune.py tests/test_full_page_omr_vocab_migration.py
git commit -m "feat: migrate OMR checkpoints by token identity"
```

---

### Task 7: Frozen Configs and Vocabulary Build Command

**Files:**
- Create: `experiments/full_page_omr/config/Page_OMR_PDMX/pretraining.json`
- Create: `experiments/full_page_omr/config/Polish_Scores/pdmx_finetuning.json`
- Create: `experiments/full_page_omr/config/Mozarteum/pdmx_finetuning.json`
- Modify: `experiments/full_page_omr/pdmx_manifest.py`
- Modify: `tests/test_full_page_omr_config.py`
- Modify: `tests/test_full_page_omr_pdmx_manifest.py`

**Interfaces:**
- Consumes: Tasks 1 through 3.
- Produces: checked-in configuration files and CLI command `build-vocabulary`.

- [ ] **Step 1: Write failing tests that load all three new configs**

```python
@pytest.mark.parametrize(
    "path",
    [
        "experiments/full_page_omr/config/Page_OMR_PDMX/pretraining.json",
        "experiments/full_page_omr/config/Polish_Scores/pdmx_finetuning.json",
        "experiments/full_page_omr/config/Mozarteum/pdmx_finetuning.json",
    ],
)
def test_new_configs_parse(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    config = experiment_config_from_dict(payload)
    if path.endswith("pretraining.json"):
        assert config.data.vocab_manifest.endswith("FullPageOMR_BeKern_v1.json")
    else:
        assert config.data.vocab_name == "FullPageOMR_BeKern_v1"
```

- [ ] **Step 2: Add the fixed PDMX config**

Use the exact JSON contract from the approved spec. `dataset_manifest` and `vocab_manifest` are repository-relative logical paths. Do not add `reduce_ratio`, `skip_steps`, or a test split.

- [ ] **Step 3: Add downstream configs without modifying legacy configs**

The Polish and Mozarteum files remain legacy Arrow-shaped configs and keep their current dataset path, batch
size, workers, tokenization, and reduce ratio. Their only vocabulary change is
`"vocab_name": "FullPageOMR_BeKern_v1"`, which makes the unchanged Arrow loader consume the generated
compatibility `.npy` pair. Do not add a second vocabulary-loading branch to the Arrow loader.

- [ ] **Step 4: Add `build-vocabulary` to the manifest CLI**

The command loads a verified dataset manifest, builds the fixed 221-token project seed, extends only from
`train_token_frequencies`, verifies validation tokens are a subset, writes
`FullPageOMR_BeKern_v1.json`, and derives its `.npy` compatibility pair through
`write_legacy_numpy_pair()`. It must refuse to overwrite an existing vocabulary or `.npy` pair with a
different digest unless `--output` names a new version.

- [ ] **Step 5: Run config and CLI tests**

Run:

```powershell
uv run pytest tests/test_full_page_omr_config.py tests/test_full_page_omr_pdmx_manifest.py tests/test_full_page_omr_vocab_manifest.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit Task 7**

```powershell
git add experiments/full_page_omr/config experiments/full_page_omr/pdmx_manifest.py tests/test_full_page_omr_config.py tests/test_full_page_omr_pdmx_manifest.py
git commit -m "feat: add frozen PDMX pretraining configs"
```

---

### Task 8: End-to-End Verification and Documentation

**Files:**
- Modify: `README.md`
- Modify: `tests/test_full_page_omr_pdmx_data.py`
- Modify: `tests/test_full_page_omr_pdmx_manifest.py`

**Interfaces:**
- Consumes: all earlier tasks.
- Produces: offline smoke command, optional official-data integration marker, and final verification evidence.

- [ ] **Step 1: Add an offline end-to-end tiny-snapshot test**

```python
def test_tiny_snapshot_scans_builds_vocab_and_yields_both_splits(tmp_path):
    snapshot = make_complete_two_renderer_snapshot(tmp_path)
    dataset_manifest = scan_fixture_snapshot(snapshot)
    vocab_manifest = build_fixture_vocabulary(dataset_manifest)
    data = build_fixture_datamodule(dataset_manifest, vocab_manifest)
    assert len(list(data.train_dataloader())) == data.steps_per_epoch
    assert len(list(data.val_dataloader())) == 2
    assert data.has_test_split is False
```

- [ ] **Step 2: Add an opt-in fixed-revision official-data smoke test**

Mark it `@pytest.mark.pdmx_official`. It calls local-only snapshot resolution and skips with a precise message when the fixed revision is absent. It reads curated validation plus one manifest-selected shard per renderer, never downloads, and asserts required fields/token coverage/tensor shapes.

- [ ] **Step 3: Document prepare, scan, vocabulary freeze, smoke train, and downstream flow**

Add commands that:

1. explicitly prepare the fixed revision;
2. scan and write the dataset manifest;
3. build and freeze the unified vocabulary;
4. launch a short `PDMX` smoke run;
5. migrate a legacy checkpoint by explicit source vocabulary;
6. fine-tune with the new Polish/Mozarteum configs.

State prominently that full PDMX training has no test split and that the official integration test never downloads implicitly.

- [ ] **Step 4: Run the focused PDMX suite**

Run:

```powershell
uv run pytest tests/test_full_page_omr_vocab_manifest.py tests/test_full_page_omr_pdmx_manifest.py tests/test_full_page_omr_pdmx_data.py tests/test_full_page_omr_vocab_migration.py -q
```

Expected: all pass.

- [ ] **Step 5: Run the complete non-network test suite**

Run:

```powershell
uv run pytest -q -m "not pdmx_official"
```

Expected: all pass. Existing unrelated failures must be reported with exact tests and must not be hidden by reducing scope.

- [ ] **Step 6: Run static repository checks**

Run:

```powershell
git diff --check
uv lock --check
```

Expected: no whitespace errors and lockfile is current.

- [ ] **Step 7: Inspect the final diff against the approved spec**

Verify:

- no changes to legacy dataset loader semantics;
- no runtime vocabulary creation;
- no validation-driven vocabulary extension;
- no fake PDMX test;
- no implicit network access in training/tests;
- no `strict=False` checkpoint loading;
- no unpinned dataset revision;
- no accidental inclusion of local cache paths or generated tar files.

- [ ] **Step 8: Commit Task 8**

```powershell
git add README.md tests/test_full_page_omr_pdmx_data.py tests/test_full_page_omr_pdmx_manifest.py
git commit -m "docs: document PDMX pretraining workflow"
```

---

## Plan Self-Review

- Spec coverage: vocabulary, dataset identity, tar validation, renderer mixing, image/token contracts, virtual epochs, honest no-test behavior, checkpoint migration, downstream configs, audit logging, and verification each map to an explicit task.
- Deliberate v1 reduction: mid-epoch exact resume is not implemented; the approved spec explicitly permits fallback to virtual-epoch-boundary full resume. The configuration and protocol advertise that limitation rather than implying stronger recovery.
- Deliberate v1 reduction: DDP and mixed real-data fine-tuning remain out of scope exactly as specified.
- Type consistency: `VocabularyManifest`, `PDMXData`, `PDMXPretrainingDataModule`, `protocol_metadata()`, and migration function names are defined once and consumed under the same names.
- Placeholder scan: every implementation task names its interfaces, failure condition, focused test command,
  and commit boundary; no deferred decision remains in an implementation step.
