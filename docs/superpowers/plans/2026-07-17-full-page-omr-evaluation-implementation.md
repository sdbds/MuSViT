# Full-page OMR Evaluation Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement canonical v2 metrics, true incremental generation, step-based validation, reproducible checkpoint diagnostics, and the `samples_seen` curriculum contract without changing the running legacy training process.

**Architecture:** Scoring is separated into immutable canonical token streams and metric views. Generation gets a dedicated state made of prepared visual memory, static per-layer cross K/V, and self K/V that preserves the configured context, while training keeps the existing full-sequence forward path. Training protocol, diagnostics, and data experiments remain explicit callers of those two foundations.

**Tech Stack:** Python 3.11, PyTorch 2.13, Lightning 2.5+, Hugging Face Transformers/Datasets, `editdistance`, `unittest`/pytest, PowerShell.

## Global Constraints

- Work only in `D:\UGit\MuSViT\.worktrees\full-page-omr-eval-v2`; do not modify executable or configuration files in the live run worktree.
- Preserve `softmax_scale=1.0`, greedy decoding, batch size 1, production `attention_window=maxlen+1`, and old checkpoint weight compatibility.
- New metrics are only `val/test_{CER,SER,LER}_v2`; checkpoint monitor is `val_SER_v2`.
- New Polish Scores baseline uses `validation_every_n_batches=10000`, `max_steps=320000`, `save_top_k=2`, and no EarlyStopping.
- Incremental generation cannot become the default until numerical tests pass and the locked RTX 4090 performance gate is measured.
- Every production edit follows RED, GREEN, REFACTOR and is committed independently.

---

### Task 1: Canonical Token Streams And V2 Metrics

**Files:**
- Rewrite: `experiments/full_page_omr/eval/eval_functions.py`
- Create: `tests/test_full_page_omr_metrics.py`

**Interfaces:**
- Produces: `CanonicalTokenStream`, `canonicalize_target_ids()`, `canonicalize_prediction_ids()`, `canonical_text()`, `metric_views()`, and `compute_canonical_metrics()`.
- Preserves: `compute_poliphony_metrics_legacy()` for one migration comparison; no new caller uses it by default.

- [ ] **Step 1: Write failing canonicalization tests**

```python
target = canonicalize_target_ids([1, 4, 5, 2, 0], I2W)
self.assertEqual(target.tokens, ("note", "<s>"))
self.assertTrue(target.terminated_by_eos)
self.assertFalse(target.truncated)

prediction = canonicalize_prediction_ids([1, 4, 0, 2], I2W, maxlen=4)
self.assertEqual(prediction.tokens, ("note", "<pad>"))
```

- [ ] **Step 2: Verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_full_page_omr_metrics.py -q`

Expected: collection fails because the canonical APIs do not exist.

- [ ] **Step 3: Implement immutable streams and adapters**

```python
@dataclass(frozen=True)
class CanonicalTokenStream:
    tokens: tuple[str, ...]
    terminated_by_eos: bool
    truncated: bool

def canonicalize_target_ids(token_ids, i2w) -> CanonicalTokenStream:
    return _canonicalize_ids(token_ids, i2w, target=True, maxlen=None)

def canonicalize_prediction_ids(token_ids, i2w, *, maxlen: int) -> CanonicalTokenStream:
    return _canonicalize_ids(token_ids, i2w, target=False, maxlen=maxlen)
```

`_canonicalize_ids()` performs one vocabulary lookup per id, removes at most one leading BOS, stops before the first EOS, preserves prediction-side PAD/BOS inside content, and raises `KeyError(f"Unknown token id {token_id}")` for an unknown id. A target without EOS raises `ValueError`; a prediction without EOS is only accepted when its raw length reaches `maxlen` and is then marked truncated.

- [ ] **Step 4: Add failing view and aggregation tests**

```python
stream = CanonicalTokenStream(("ab", "<s>", "c", "<t>", "d", "<b>"), True, False)
self.assertEqual(metric_views(stream).cer, tuple("ab c\td\n"))
self.assertEqual(metric_views(stream).ser, ("ab", "c", "<t>", "d", "<b>"))
self.assertEqual(metric_views(stream).ler, ("ab c\td",))
```

Cover Unicode `·`, empty ground-truth totals, split-level micro-averaging, and parameterized equality between `editdistance.eval` and the legacy dynamic program.

- [ ] **Step 5: Implement views and metrics with `editdistance.eval`**

```python
@dataclass(frozen=True)
class MetricViews:
    cer: tuple[str, ...]
    ser: tuple[str, ...]
    ler: tuple[str, ...]

def compute_canonical_metrics(predictions, targets) -> tuple[float, float, float]:
    prediction_views = tuple(metric_views(stream) for stream in predictions)
    target_views = tuple(metric_views(stream) for stream in targets)
    return tuple(_micro_error_rate(prediction_views, target_views, name) for name in ("cer", "ser", "ler"))
```

- [ ] **Step 6: Verify GREEN and commit**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_full_page_omr_metrics.py -q`

Commit: `feat: add canonical full-page OMR metrics`

### Task 2: Raw Generation Results And Trainer Metric Integration

**Files:**
- Modify: `experiments/full_page_omr/smt_foundation/modeling_smt.py`
- Modify: `experiments/full_page_omr/smt_trainer.py`
- Modify: `tests/test_full_page_omr_model_contracts.py`
- Create: `tests/test_full_page_omr_generation.py`

**Interfaces:**
- Produces: immutable `GenerationResult(token_ids, output, terminated_by_eos, truncated)` and `generate_token_ids(input, use_incremental=False)`.
- Consumes: Task 1 canonical adapters and metrics.
- Preserves: `predict()` returns `(list[str], SMTOutput)` and still accepts deprecated `convert_to_str`.

- [ ] **Step 1: Write failing generation-result compatibility tests**

```python
result = SMTFoundationModelForCausalLM.generate_token_ids(model, image, use_incremental=False)
self.assertEqual(result.token_ids, (0, 1, 2))
self.assertTrue(result.terminated_by_eos)
self.assertFalse(result.truncated)
self.assertEqual(SMTFoundationModelForCausalLM.predict(model, image)[0], ["note"])
```

- [ ] **Step 2: Verify RED, then implement uncached raw-id generation**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_full_page_omr_generation.py tests/test_full_page_omr_model_contracts.py -q`

Implement `GenerationResult` and make `predict()` a compatibility adapter over the uncached raw-id generator.

- [ ] **Step 3: Write failing trainer metric tests**

Use a tiny model returning raw ids `(bos, note, eos)` and a target tensor `(note, eos, pad)`. Assert `validation_step()` stores canonical streams and `on_validation_epoch_end()` logs exactly `val_CER_v2`, `val_SER_v2`, and `val_LER_v2`.

- [ ] **Step 4: Replace string reconstruction in `SMTPP_Trainer`**

```python
generation = self.model.generate_token_ids(input=x)
self.preds.append(canonicalize_prediction_ids(generation.token_ids, self.model.i2w, maxlen=self.model.maxlen))
self.grtrs.append(canonicalize_target_ids(y.squeeze(0).tolist(), self.model.i2w))
```

`on_validation_epoch_end()` calls `compute_canonical_metrics()`, logs only versioned names, and clears both buffers even if display sampling is disabled.

- [ ] **Step 5: Verify GREEN and commit**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_full_page_omr_metrics.py tests/test_full_page_omr_generation.py tests/test_full_page_omr_model_contracts.py tests/test_full_page_omr_throughput.py -q`

Commit: `feat: score canonical full-page OMR predictions`

### Task 3: Projected Attention And True Incremental Decoder

**Files:**
- Modify: `experiments/full_page_omr/smt_foundation/modeling_smt.py`
- Modify: `tests/test_smt_attention.py`
- Modify: `tests/test_full_page_omr_generation.py`

**Interfaces:**
- Produces: `AttentionKV`, `GenerationMemory`, `DecoderLayerGenerationState`, `DecoderGenerationState`, `MHA.project_key_value()`, `MHA.forward_projected()`, `Decoder.prepare_generation_memory()`, `Decoder.init_generation_state()`, and `Decoder.decode_step()`.
- Consumes: Task 2 `GenerationResult`.
- Preserves: full-sequence `MHA.forward()`, `Decoder.forward()`, loss shapes, and state-dict keys.

- [ ] **Step 1: Write failing projected-attention equivalence test**

```python
kv = attention.project_key_value(key, value)
actual = attention.forward_projected(query, kv, get_weights=False)
expected = attention(query, key, value, get_weights=False)
torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
```

- [ ] **Step 2: Verify RED, then extract projection from `MHA.forward()`**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_smt_attention.py -q`

`AttentionKV.key/value` use `[batch, heads, source, head_dim]`; `forward_projected()` projects only Q and routes through the existing FlashAttention/SDPA/eager selection without changing scale or mask semantics.

- [ ] **Step 3: Write failing incremental-equivalence and window tests**

```python
for position in range(tokens.size(1)):
    step_logits, state = decoder.decode_step(memory, tokens[:, position:position + 1], state)
    full_logits = full_decoder_logits(tokens[:, :position + 1])
    torch.testing.assert_close(step_logits, full_logits, rtol=1e-5, atol=1e-5)
    self.assertEqual(state.layers[0].self_kv.key.size(2), position + 1)
```

Patch each layer's cross `lk/lv` and assert they are each called once in `prepare_generation_memory()` and never in `decode_step()`. Add a separate decoder with `attention_window=4` and assert its write-back cache never exceeds 3 historical positions.

- [ ] **Step 4: Implement context-preserving per-layer generation state**

```python
@dataclass(frozen=True)
class DecoderGenerationState:
    position: int
    layers: tuple[DecoderLayerGenerationState, ...]

def decode_step(self, memory, token, state):
    # embed only token, apply absolute PE at state.position, append current self K/V,
    # preserve all history when attention_window exceeds maxlen; only a smaller explicit
    # window keeps at most attention_window - 1 entries for the next step.
```

- [ ] **Step 5: Prepare visual memory once and add incremental raw generation**

Extract adaptor/2D PE/flatten into `prepare_generation_memory()`. `generate_token_ids(input=image, use_incremental=True)` uses it once and then calls `decode_step()` until EOS or `maxlen`.

- [ ] **Step 6: Verify GREEN and commit**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_smt_attention.py tests/test_full_page_omr_generation.py tests/test_full_page_omr_model_contracts.py -q`

Commit: `feat: add incremental OMR decoding`

### Task 4: Step-Based Validation And Versioned Checkpoints

**Files:**
- Modify: `2.full_page_omr.ps1`
- Modify: `experiments/full_page_omr/entrypoint.py`
- Modify: `experiments/full_page_omr/finetune.py`
- Modify: `tests/test_full_page_omr_throughput.py`

**Interfaces:**
- Produces: `_validate_validation_every_n_batches()`, `_validate_max_steps()`, and `_build_metric_checkpointer()`.
- Removes: EarlyStopping callback from new runs.

- [ ] **Step 1: Write failing schedule and callback tests**

Assert all public layers default to `validation_every_n_batches=10000` and `max_steps=320000`; reject bool, zero, negative validation intervals, and `max_steps=-1` when `train=True`. Assert metric checkpoint filename contains `{step}` and `{val_SER_v2:.4f}`, monitor is `val_SER_v2`, and `save_top_k == 2`.

- [ ] **Step 2: Verify RED, then implement validators and callback builder**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_full_page_omr_throughput.py -q`

```python
Trainer(
    max_steps=max_steps,
    check_val_every_n_epoch=None,
    val_check_interval=validation_every_n_batches,
    callbacks=[epoch_checkpointer, metric_checkpointer],
    precision="16-mixed",
)
```

- [ ] **Step 3: Forward the protocol through PowerShell, entrypoint, launch, and main**

PowerShell emits `--validation_every_n_batches=10000` and `--max_steps=320000`. W&B hyperparameters record `protocol_version=full_page_omr_eval_v2` and `metric_version=canonical_v2` with all schedule fields.

- [ ] **Step 4: Verify GREEN and commit**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_full_page_omr_throughput.py tests/test_full_page_omr_config.py -q`

Commit: `feat: add versioned OMR validation protocol`

### Task 5: Checkpointed `samples_seen` Curriculum Position

**Files:**
- Modify: `experiments/full_page_omr/smt_trainer.py`
- Modify: `experiments/full_page_omr/data.py`
- Modify: `experiments/full_page_omr/finetune.py`
- Modify: `tests/test_full_page_omr_data_pipeline.py`
- Modify: `tests/test_full_page_omr_throughput.py`

**Interfaces:**
- Produces: `SMTPP_Trainer.samples_seen`, `curriculum_step`, checkpoint key `full_page_omr_samples_seen`, and data-module reset from the restored module.
- Preserves: `skip_steps` as the source-run offset; new run-local `samples_seen` starts at zero.

- [ ] **Step 1: Write failing save/load and consumption tests**

```python
module.training_step(batch)
self.assertEqual(module.samples_seen, 1)
checkpoint = {}
module.on_save_checkpoint(checkpoint)
self.assertEqual(checkpoint["full_page_omr_samples_seen"], 1)
```

Cover legacy inference from `global_step` only for batch size 1 and accumulation 1, rejection otherwise, and no increment when forward raises.

- [ ] **Step 2: Verify RED, then implement the run-local counter**

`training_step()` synchronizes encoder trainability from `skip_steps + samples_seen`, computes loss, then increments by `x.shape[0]`. `on_load_checkpoint()` validates a non-negative integer counter or performs the constrained legacy migration with a warning.

- [ ] **Step 3: Write failing data-loader reset tests**

Set `module.trainer.lightning_module.samples_seen=23` and `skip_steps=120000`; assert the first reserved step after `train_dataloader()` is 120023, independent of optimizer `global_step`.

- [ ] **Step 4: Implement data-module reset and verify GREEN**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_full_page_omr_data_pipeline.py tests/test_full_page_omr_throughput.py -q`

Commit: `feat: checkpoint OMR curriculum sample position`

### Task 6: Reproducible Checkpoint Diagnostics

**Files:**
- Create: `experiments/full_page_omr/diagnose_checkpoints.py`
- Create: `tests/test_full_page_omr_diagnostics.py`

**Interfaces:**
- Produces: `load_manifest()`, `verify_checkpoint_identity()`, `encoder_state_sha256()`, `relative_encoder_drift()`, and a Fire-compatible `main(manifest_path, output_path)`.
- Consumes: canonical metrics and raw generation from Tasks 1-3.

- [ ] **Step 1: Write failing manifest identity tests**

Create tiny temporary checkpoints and assert mismatched SHA-256, epoch, or global step fails before model loading. Assert missing pre checkpoint yields `status="partial"` only in post-only mode and fails in pre/post mode.

- [ ] **Step 2: Implement strict manifest parsing and file hashing**

Require foundation model/revision/state digest, dataset revision, fixed val protocol, checkpoint absolute path, SHA-256, epoch, step, and either run id or non-empty source note.

- [ ] **Step 3: Write failing deterministic drift tests**

```python
reference = {"block.weight": torch.tensor([3.0, 4.0])}
candidate = {"block.weight": torch.tensor([0.0, 0.0])}
self.assertEqual(relative_encoder_drift(reference, candidate)["global"], 1.0)
```

Cover sorted-name state digest, missing/extra keys, shape mismatch, float64 accumulation, and per-block results.

- [ ] **Step 4: Implement drift and structured report assembly**

The CLI writes JSON containing the validated input manifest, exact checkpoint identity, per-page v2 metrics, aggregate metrics, EOS/truncation counts, decode time, and global/per-block drift. Test evaluation is dependency-injected so unit tests do not download data or load the 1.1 GB checkpoint.

- [ ] **Step 5: Verify GREEN and commit**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_full_page_omr_diagnostics.py tests/test_full_page_omr_metrics.py -q`

Commit: `feat: add reproducible OMR checkpoint diagnostics`

### Task 7: Single-Resize Experiment And Vocabulary Failure Semantics

**Files:**
- Modify: `experiments/full_page_omr/data.py`
- Create: `experiments/full_page_omr/config/Polish_Scores/finetuning_single_resize.json`
- Modify: `experiments/full_page_omr/utils/vocab_utils.py`
- Modify: `tests/test_full_page_omr_data_pipeline.py`
- Create: `tests/test_full_page_omr_vocab.py`

**Interfaces:**
- Produces: an isolated Polish Scores config with `reduce_ratio=1.0`.
- Preserves: baseline Polish config at `reduce_ratio=0.5` and all Mozarteum behavior.

- [ ] **Step 1: Write failing resize tests**

Patch `cv2.resize`: ratio 0.5 must resize once to the intermediate dimensions; ratio 1.0 must return the original array without calling OpenCV before the common 1024 transform.

- [ ] **Step 2: Implement identity-ratio bypass and experiment config**

```python
if self.reduce_ratio != 1.0:
    image = cv2.resize(image, (width, height))
```

- [ ] **Step 3: Write failing vocabulary exception test and remove swallowing decorator**

Patch `np.load` to raise `OSError("broken vocab")`; assert `check_and_retrieveVocabulary()` raises that exact source exception. Remove `@logger.catch` instead of returning `None`.

- [ ] **Step 4: Verify GREEN and commit**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_full_page_omr_data_pipeline.py tests/test_full_page_omr_vocab.py -q`

Commit: `feat: isolate Polish Scores resize experiment`

### Task 8: Locked Generation Benchmark And Full Verification

**Files:**
- Create: `experiments/full_page_omr/benchmark_generation.py`
- Create: `tests/test_full_page_omr_generation_benchmark.py`
- Modify: `docs/superpowers/specs/2026-07-17-full-page-omr-evaluation-and-accuracy-design.md` only to record measured status, not to change thresholds.

**Interfaces:**
- Produces: a JSON report containing environment identity, locked checkpoint/dataset/GPU identity, token equality, 1024/2048/4096 median decoder timings, complete 10-page val timings, and gate status.
- Consumes: Task 3 uncached and incremental generation paths.

- [ ] **Step 1: Write failing deterministic prefix and timing-summary tests**

Assert row 5 content tokens are cycled after BOS to exact requested lengths. Mock `perf_counter` and CUDA synchronization to assert 3 warmups/10 samples for microbenchmarks and 1 warmup/3 samples for full val.

- [ ] **Step 2: Implement benchmark harness with locked constants**

Hard-code the approved checkpoint SHA, dataset revision, RTX 4090 UUID, float16 dtype, resolution 1024, reduce ratio 0.5, maxlen 7512, and `attention_backend=auto`. Identity mismatch returns `status="not-run"`; it never substitutes a different sample or device.

- [ ] **Step 3: Verify harness tests and run the CPU numerical suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_full_page_omr_generation_benchmark.py tests/test_smt_attention.py tests/test_full_page_omr_generation.py -q`

- [ ] **Step 4: Run the locked GPU benchmark only after the live run releases the specified RTX 4090**

If the GPU or reference checkpoint is unavailable, write a `not-run` report and keep uncached generation as default. If all numerical and performance gates pass, switch the production default in one final tested commit.

- [ ] **Step 5: Run complete verification and commit**

Run: `.\.venv\Scripts\python.exe -m pytest tests -q`

Run: `git diff --check`

Commit: `test: lock full-page OMR generation benchmark`
