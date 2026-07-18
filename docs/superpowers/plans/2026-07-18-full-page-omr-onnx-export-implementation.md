# Full-Page OMR ONNX Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a standalone two-graph ONNX Runtime deployment for the first Polish Scores CL full-page OMR checkpoint.

**Architecture:** Construct the complete PyTorch model offline from embedded OMR and ViT configs, then strictly load the inference safetensors. Export a fixed-resolution visual graph and a dynamic-prefix decoder graph; keep greedy EOS control and vocabulary decoding in a small ONNX Runtime host helper.

**Tech Stack:** Python 3.11, PyTorch 2.13, Transformers 4.57, safetensors 0.7, ONNX opset 20, ONNX Runtime GPU 1.24.4+, pytest, uv.

## Global Constraints

- Install `onnx` and `onnxruntime-gpu>=1.24.4` directly into `D:\UGit\MuSViT\.venv` with uv; do not create another environment.
- Preserve all 413 FP32 checkpoint tensors and require strict loading with zero missing and zero unexpected keys.
- Do not read or download foundation-model weights during standalone construction or ONNX Runtime inference.
- Keep batch size fixed at 1 and page input fixed at float32 `[1, 3, 1024, 1024]` in `[0, 1]`.
- Export `encoder.onnx` outputs `raw_features` and `enhanced_features`, both float32 `[1, 4096, 256]`.
- Export `decoder.onnx` with dynamic complete-prefix input `token_ids` shaped int64 `[1, T]` and output `next_token_logits` shaped float32 `[1, 215]`.
- Preserve uncached full-prefix semantics; do not export the benchmark-only KV-cache path or an ONNX `Loop`.
- Use BOS id 100, EOS id 183, maximum length 7512, opset 20, FP32, `rtol=1e-4`, and `atol=1e-4`.
- Require exact greedy token-id parity on Polish Scores validation row 0 at revision `b3170c8b8f322885b566efe9e264af9328b5603f`.
- Keep each graph and the combined graph size below 2 GiB and reject external tensor-data sidecars.
- Do not modify the untracked root `onnx_export.py` or unrelated dirty worktree files.

---

## File Structure

- Modify `experiments/full_page_omr/smt_foundation/configuration_smt.py`: retain an optional embedded foundation encoder config.
- Modify `experiments/full_page_omr/smt_foundation/modeling_smt.py`: construct the ViT encoder from embedded config without loading remote weights.
- Create `experiments/full_page_omr/export_onnx.py`: load the standalone model, define graph wrappers, export atomically, write metadata, and verify PyTorch/ORT parity.
- Create `experiments/full_page_omr/onnx_runtime.py`: preprocess pages, create sessions, run the two graphs, and perform host-side greedy decoding.
- Create `tests/test_full_page_omr_onnx.py`: cover offline construction, graph contracts, export options, runtime orchestration, and failure modes.
- Generate ignored assets under `experiments/full_page_omr/weights/polish_scores_cl_CL_onnx/`; do not add ONNX binaries to git.

### Task 1: Install And Verify The Export Runtime

**Files:**
- Environment: `D:\UGit\MuSViT\.venv`
- Reference: `D:\musubi-tuner-scripts\qinglong-captions\pyproject.toml`

**Interfaces:**
- Consumes: the existing project virtual environment with PyTorch, Transformers, and safetensors.
- Produces: importable `onnx` and `onnxruntime` packages in the same environment.

- [ ] **Step 1: Record the pre-install package state**

Run:

```powershell
uv pip show --python .venv\Scripts\python.exe onnx onnxruntime-gpu
```

Expected: both packages are reported as missing before installation.

- [ ] **Step 2: Install ONNX from PyPI with uv**

Run:

```powershell
uv pip install --python .venv\Scripts\python.exe onnx
```

Expected: `onnx` installs successfully without replacing the locked Torch build.

- [ ] **Step 3: Install the CUDA 13 ONNX Runtime build with uv**

Run:

```powershell
uv pip install --python .venv\Scripts\python.exe --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-13/pypi/simple/ --extra-index-url https://pypi.org/simple/ "onnxruntime-gpu>=1.24.4"
```

Expected: the package resolves from the same CUDA 13 index used by qinglong-captions.

- [ ] **Step 4: Verify versions and CUDA provider discovery**

Run:

```powershell
.venv\Scripts\python.exe -c "import onnx, onnxruntime as ort, torch; print({'onnx': onnx.__version__, 'ort': ort.__version__, 'torch': torch.__version__, 'providers': ort.get_available_providers()})"
```

Expected: ONNX is importable, ORT is at least 1.24.4, Torch remains 2.13.0+cu130, and `CUDAExecutionProvider` is listed.

### Task 2: Add Offline Foundation Construction

**Files:**
- Modify: `experiments/full_page_omr/smt_foundation/configuration_smt.py`
- Modify: `experiments/full_page_omr/smt_foundation/modeling_smt.py`
- Test: `tests/test_full_page_omr_onnx.py`

**Interfaces:**
- Consumes: `SMTFoundationConfig.foundation_architecture`, optional `foundation_config: dict[str, Any] | None`, and the existing online `foundation_weights` fallback.
- Produces: `_build_foundation_encoder(config: SMTFoundationConfig) -> torch.nn.Module` and an unchanged public `SMTFoundationModelForCausalLM(config)` constructor.

- [ ] **Step 1: Write failing config and offline-construction tests**

Add tests equivalent to:

```python
def test_config_retains_embedded_foundation_config():
    embedded = {"hidden_size": 768, "num_hidden_layers": 1}
    config = SMTFoundationConfig(foundation_config=embedded)
    assert config.foundation_config == embedded


def test_model_uses_embedded_config_without_from_pretrained(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("from_pretrained must not be called")

    monkeypatch.setattr(ViTModel, "from_pretrained", fail_if_called)
    config = make_small_offline_smt_config()
    model = SMTFoundationModelForCausalLM(config)
    assert model.encoder.config.image_size == 32
    assert model.encoder.config.hidden_size == 768
```

`make_small_offline_smt_config()` uses image size 32, patch size 16, hidden size 768, one ViT layer, 12 heads, decoder max length 8, and five output categories so the test remains lightweight while exercising the real constructor.

- [ ] **Step 2: Run the tests and verify the expected failure**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_full_page_omr_onnx.py -k "foundation_config or embedded_config" -q
```

Expected: FAIL because `foundation_config` is not retained and the constructor still calls `from_pretrained`.

- [ ] **Step 3: Implement the offline construction branch**

Add `foundation_config=None` to `SMTFoundationConfig.__init__` and assign it to `self.foundation_config`. In `modeling_smt.py`, import `ViTConfig` and centralize encoder creation:

```python
def _build_foundation_encoder(config):
    raw_config = getattr(config, "foundation_config", None)
    if raw_config is not None:
        if not isinstance(raw_config, dict):
            raise TypeError("foundation_config must be a dictionary")
        encoder_config = ViTConfig.from_dict(dict(raw_config))
        return ViTModel(encoder_config)

    encoder_class = IMPL_DICT[config.foundation_architecture][0]
    if config.foundation_architecture == "ViTMAEBase":
        return encoder_class.from_pretrained(
            config.foundation_weights,
            mask_ratio=0.0,
        )
    return encoder_class.from_pretrained(config.foundation_weights)
```

Replace the constructor's existing conditional with `self.encoder = _build_foundation_encoder(config)`. Preserve the old online fallback exactly when no embedded config is supplied.

- [ ] **Step 4: Run focused and model-contract tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_full_page_omr_onnx.py tests/test_full_page_omr_model_contracts.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the offline-construction change**

```powershell
git add experiments/full_page_omr/smt_foundation/configuration_smt.py experiments/full_page_omr/smt_foundation/modeling_smt.py tests/test_full_page_omr_onnx.py
git commit -m "feat: construct full-page OMR encoder offline"
```

### Task 3: Implement Exact Encoder And Decoder Export Wrappers

**Files:**
- Create: `experiments/full_page_omr/export_onnx.py`
- Modify: `tests/test_full_page_omr_onnx.py`

**Interfaces:**
- Consumes: a strictly loaded `SMTFoundationModelForCausalLM`.
- Produces: `FullPageOMREncoderWrapper.forward(pixel_values) -> tuple[Tensor, Tensor]`, `FullPageOMRDecoderWrapper.forward(raw_features, enhanced_features, token_ids) -> Tensor`, and `load_standalone_model(weights_path, model_config_path, encoder_config_path) -> tuple[SMTFoundationModelForCausalLM, dict]`.

- [ ] **Step 1: Write failing strict-load and wrapper-parity tests**

Cover these contracts:

```python
def test_encoder_wrapper_returns_batch_first_prepared_features():
    raw, enhanced = FullPageOMREncoderWrapper(fake_model)(pixel_values)
    assert raw.shape == (1, 4, 16)
    assert enhanced.shape == (1, 4, 16)


def test_decoder_wrapper_matches_uncached_full_prefix_reference():
    actual = FullPageOMRDecoderWrapper(fake_model)(raw, enhanced, token_ids)
    expected = reference_full_prefix_last_logits(fake_model.decoder, raw, enhanced, token_ids)
    torch.testing.assert_close(actual, expected)


def test_standalone_loader_rejects_missing_or_unexpected_state_keys(tmp_path):
    with pytest.raises(RuntimeError, match="state_dict"):
        load_standalone_model(bad_weights, model_config, encoder_config)
```

- [ ] **Step 2: Run wrapper tests and verify they fail**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_full_page_omr_onnx.py -k "wrapper or standalone_loader" -q
```

Expected: FAIL because the export module and wrappers do not exist.

- [ ] **Step 3: Implement strict standalone loading**

`load_standalone_model()` must:

1. parse both JSON configs with structured JSON APIs;
2. embed the encoder config into the OMR config before construction;
3. construct on CPU without `from_pretrained`;
4. load safetensors on CPU;
5. call `load_state_dict(..., strict=True)`;
6. force both model and decoder attention backends to `eager` for an explicit ONNX graph;
7. return the eval model and the merged standalone config dictionary.

- [ ] **Step 4: Implement the encoder wrapper**

The wrapper must call the model encoder once, apply `_prepare_decoder_features()`, and expose batch-first outputs:

```python
class FullPageOMREncoderWrapper(torch.nn.Module):
    def forward(self, pixel_values):
        encoder_output = self.model.forward_encoder(pixel_values)
        prepared = self.model._prepare_decoder_features(
            encoder_output.permute(0, 2, 1).contiguous()
        )
        return (
            prepared.raw_features.permute(1, 0, 2).contiguous(),
            prepared.enhanced_features.permute(1, 0, 2).contiguous(),
        )
```

- [ ] **Step 5: Implement the exact full-prefix decoder wrapper**

The wrapper converts the batch-first visual inputs back to sequence-first, embeds the entire prefix, applies 1D positions, runs every decoder layer over every prefix position with causal attention and no cache, projects only `output[-1:]`, and returns `[1, 215]`. It must not call `decode_step()` or pass `predict_last_n_only` into intermediate decoder layers.

- [ ] **Step 6: Run focused wrapper tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_full_page_omr_onnx.py -k "wrapper or standalone_loader" -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit wrappers and strict loader**

```powershell
git add experiments/full_page_omr/export_onnx.py tests/test_full_page_omr_onnx.py
git commit -m "feat: add full-page OMR ONNX graph wrappers"
```

### Task 4: Add Atomic ONNX Export And Bundle Metadata

**Files:**
- Modify: `experiments/full_page_omr/export_onnx.py`
- Modify: `tests/test_full_page_omr_onnx.py`

**Interfaces:**
- Consumes: the wrappers from Task 3 and explicit input/output paths.
- Produces: `ExportPaths`, `export_encoder_graph()`, `export_decoder_graph()`, `write_bundle_metadata()`, `validate_graph_files()`, `export_bundle()`, and a CLI `main()`.

- [ ] **Step 1: Write failing export-contract tests**

Assert that mocked `torch.onnx.export` calls use opset 20, `dynamo=False`, fixed encoder axes, a dynamic decoder token axis named `sequence_length`, exact graph input/output names, and temporary output paths. Add tests that reject a zero-byte graph, an unexpected `.data` sidecar, and a graph at or above 2 GiB.

- [ ] **Step 2: Run export-contract tests and verify failure**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_full_page_omr_onnx.py -k "export or graph_file or metadata" -q
```

Expected: FAIL because export and metadata functions are absent.

- [ ] **Step 3: Implement atomic graph export**

Use `torch.onnx.export(..., opset_version=20, dynamo=False, do_constant_folding=True)` with:

```python
encoder_input_names = ["pixel_values"]
encoder_output_names = ["raw_features", "enhanced_features"]
decoder_input_names = ["raw_features", "enhanced_features", "token_ids"]
decoder_output_names = ["next_token_logits"]
decoder_dynamic_axes = {"token_ids": {1: "sequence_length"}}
```

Write each graph to `<name>.onnx.tmp`, run `onnx.checker.check_model()`, and use `os.replace()` only after validation succeeds. Explicitly disable external data for these sub-2-GiB graphs using the PyTorch 2.13 exporter argument supported by the installed signature.

- [ ] **Step 4: Implement metadata and config output**

Write atomically:

- `config.json` containing the OMR config plus embedded encoder config and vocabulary;
- `preprocessor_config.json` recording RGB, direct 1024 x 1024 bilinear resize, rescale factor `1/255`, and no normalization;
- `metadata.json` recording source/config hashes, graph contracts, BOS/EOS/max length, opset, package versions, graph hashes/sizes, providers, and validation fields.

- [ ] **Step 5: Implement CLI and dry plan output**

Support explicit `--weights-path`, `--model-config-path`, `--encoder-config-path`, `--output-dir`, `--device`, `--opset-version`, `--verify-runtime`, and `--verify-dataset` arguments. Defaults target the first checkpoint and `polish_scores_cl_CL_onnx` output directory. Print the resolved plan before loading tensors.

- [ ] **Step 6: Run export tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_full_page_omr_onnx.py -k "export or graph_file or metadata or cli" -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit the exporter**

```powershell
git add experiments/full_page_omr/export_onnx.py tests/test_full_page_omr_onnx.py
git commit -m "feat: export standalone full-page OMR ONNX bundle"
```

### Task 5: Implement The ONNX Runtime Greedy Host

**Files:**
- Create: `experiments/full_page_omr/onnx_runtime.py`
- Modify: `tests/test_full_page_omr_onnx.py`

**Interfaces:**
- Consumes: the generated bundle and optional ONNX Runtime provider list.
- Produces: `preprocess_page(image, config) -> numpy.ndarray`, `OnnxGenerationResult`, and `FullPageOMROnnxRuntime.generate_pixel_values(pixel_values) -> OnnxGenerationResult` plus `generate(image)`.

- [ ] **Step 1: Write failing preprocessing and fake-session tests**

Test direct RGB conversion, bilinear resize, NCHW float32 `[0, 1]`, encoder output routing, complete-prefix growth `[100] -> [100, token]`, EOS termination at 183, max-length truncation, vocabulary conversion, and rejection of malformed session contracts.

- [ ] **Step 2: Run runtime tests and verify failure**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_full_page_omr_onnx.py -k "preprocess or runtime or greedy" -q
```

Expected: FAIL because the runtime module does not exist.

- [ ] **Step 3: Implement provider and session validation**

Prefer `CUDAExecutionProvider` when requested and available, append `CPUExecutionProvider` as fallback, and reject missing graph input/output names before inference. Keep `onnxruntime` imports inside runtime-facing functions so model/export unit tests remain importable without ORT.

- [ ] **Step 4: Implement preprocessing and graph calls**

Use Pillow `Image.Resampling.BILINEAR`, NumPy float32 conversion divided by 255, CHW transpose, and batch insertion. Ensure contiguous arrays before `session.run()`.

- [ ] **Step 5: Implement exact host-side greedy decoding**

Start with `np.asarray([[bos_token_id]], dtype=np.int64)`, send the complete growing prefix on every decoder call, use `np.argmax(logits, axis=1)`, stop at EOS, and return immutable token ids, decoded tokens excluding BOS/EOS, termination, and truncation fields.

- [ ] **Step 6: Run runtime tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_full_page_omr_onnx.py -k "preprocess or runtime or greedy" -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit the runtime helper**

```powershell
git add experiments/full_page_omr/onnx_runtime.py tests/test_full_page_omr_onnx.py
git commit -m "feat: add full-page OMR ONNX Runtime decoding"
```

### Task 6: Export And Verify The Real Checkpoint

**Files:**
- Input: `experiments/full_page_omr/weights/polish_scores_cl_CL.safetensors`
- Input: `experiments/full_page_omr/weights/polish_scores_cl_CL.config.json`
- Input config source: cached foundation revision `eecd5b327521225e65e1c2fe38ab99eb667c1609/config.json`
- Generate ignored bundle: `experiments/full_page_omr/weights/polish_scores_cl_CL_onnx/`

**Interfaces:**
- Consumes: Tasks 1-5.
- Produces: checked `encoder.onnx`, `decoder.onnx`, and three companion JSON files.

- [ ] **Step 1: Run the complete focused test module**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_full_page_omr_onnx.py -q
```

Expected: all tests pass before the expensive real export.

- [ ] **Step 2: Export the real bundle with runtime tensor verification**

Run:

```powershell
.venv\Scripts\python.exe -m experiments.full_page_omr.export_onnx --weights-path experiments/full_page_omr/weights/polish_scores_cl_CL.safetensors --model-config-path experiments/full_page_omr/weights/polish_scores_cl_CL.config.json --encoder-config-path C:\Users\qingl\.cache\huggingface\hub\models--carlospm12--LSMT-MAE-Base-1024-16\snapshots\eecd5b327521225e65e1c2fe38ab99eb667c1609\config.json --output-dir experiments/full_page_omr/weights/polish_scores_cl_CL_onnx --device cuda --verify-runtime
```

Expected: both graphs export, pass ONNX checker, load in ORT CUDA and CPU sessions, satisfy `rtol=atol=1e-4`, and produce matching checked-prefix argmax ids.

- [ ] **Step 3: Run the locked real-page parity gate**

Run the same command with `--verify-dataset`. Expected: PyTorch and ONNX Runtime produce the same complete token-id sequence, EOS status, and decoded tokens for Polish Scores validation row 0 at the locked revision.

- [ ] **Step 4: Audit graph sizes, hashes, sidecars, and metadata**

Run:

```powershell
Get-ChildItem experiments\full_page_omr\weights\polish_scores_cl_CL_onnx -Force | Select-Object Name,Length
Get-FileHash experiments\full_page_omr\weights\polish_scores_cl_CL_onnx\encoder.onnx -Algorithm SHA256
Get-FileHash experiments\full_page_omr\weights\polish_scores_cl_CL_onnx\decoder.onnx -Algorithm SHA256
```

Expected: exactly two `.onnx` files and three JSON files, no `.data` sidecars or temporary files, each graph and combined ONNX size below 2 GiB, and metadata hashes match the files.

### Task 7: Run Regression Verification And Final Review

**Files:**
- Verify: all changed source and tests.
- Preserve: unrelated dirty files and ignored model artifacts.

**Interfaces:**
- Consumes: the completed implementation and real bundle.
- Produces: fresh evidence that existing model, attention, generation, and new ONNX behavior all pass together.

- [ ] **Step 1: Run focused regression tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_full_page_omr_onnx.py tests/test_full_page_omr_model_contracts.py tests/test_full_page_omr_generation.py tests/test_smt_attention.py -q
```

Expected: zero failures.

- [ ] **Step 2: Run syntax and whitespace checks**

Run:

```powershell
.venv\Scripts\python.exe -m compileall experiments/full_page_omr/export_onnx.py experiments/full_page_omr/onnx_runtime.py experiments/full_page_omr/smt_foundation
git diff --check
```

Expected: compilation succeeds and git reports no whitespace errors in tracked changes.

- [ ] **Step 3: Review the final diff and worktree boundaries**

Run:

```powershell
git status --short
git diff --stat
git log -5 --oneline
```

Expected: only task-owned source/test changes are committed; pre-existing edits to `2.full_page_omr.ps1`, `finetune.py`, `tests/test_full_page_omr_throughput.py`, and untracked `onnx_export.py` remain untouched.

- [ ] **Step 4: Record final artifact evidence**

Report ONNX and ORT versions, providers, graph sizes and SHA-256 hashes, numerical maximum errors, real-page token count/EOS parity, test counts, and any provider-specific limitation. Do not claim completion without fresh outputs from Tasks 6 and 7.
