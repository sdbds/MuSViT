# Full-Page OMR ONNX Export Design

## Context

The first Polish Scores CL checkpoint has been reduced from a PyTorch Lightning
training checkpoint to two inference assets:

- `experiments/full_page_omr/weights/polish_scores_cl_CL.safetensors`
- `experiments/full_page_omr/weights/polish_scores_cl_CL.config.json`

The tensor payload is complete. A strict load into a freshly constructed model
accounts for all 413 state entries with no missing or unexpected keys:

| Component | Tensor count | FP32 payload |
| --- | ---: | ---: |
| ViT encoder | 200 | 340.97 MiB |
| Adaptor | 2 | 0.75 MiB |
| Autoregressive decoder | 211 | 20.55 MiB |

The existing model constructor calls `ViTModel.from_pretrained()` to obtain the
encoder structure and initial values. This is an initialization dependency, not
a missing-weight dependency: every encoder tensor is already present in the
inference safetensors file. The external model named by the checkpoint is the
vision encoder, not an additional OMR decoder.

## Goals

1. Produce a self-contained ONNX Runtime deployment of the complete full-page
   OMR model without downloading any foundation-model weights at runtime.
2. Split the graph after visual feature preparation so the 1024 x 1024 image is
   encoded exactly once per page.
3. Preserve the current production uncached greedy-decoding semantics: every
   decoder invocation consumes the complete token prefix.
4. Keep the combined ONNX payload below 2 GiB.
5. Verify graph validity, ONNX Runtime execution, tensor-level numerical parity,
   and greedy token parity against PyTorch.

## Non-Goals

- Do not export the benchmark-only incremental KV-cache path. Its locked FP16
  benchmark produced a real token divergence on one of ten validation pages and
  did not pass the performance gate.
- Do not quantize or cast the checkpoint to FP16/BF16. The first export retains
  FP32 checkpoint fidelity.
- Do not embed greedy generation in an ONNX `Loop` node.
- Do not change or extend the untracked root `onnx_export.py`; it remains a
  behavioral reference for opset, metadata, preprocessing, and verification.
- Do not integrate the new transcription runtime into qinglong-captions in this
  task.

## Chosen Architecture

### Offline model construction

The safetensors payload contains all learned values, but the existing OMR config
does not contain the full non-tensor ViT architecture contract. Exact offline
construction also needs the locked foundation encoder config, including image
and patch sizes, layer count, attention dimensions, activation, and LayerNorm
epsilon.

The export flow therefore takes an explicit encoder config and embeds it into
the standalone model config. For this checkpoint the config comes from the
locally cached locked snapshot:

```text
carlospm12/LSMT-MAE-Base-1024-16
revision eecd5b327521225e65e1c2fe38ab99eb667c1609
```

No foundation weight file is read. The exporter constructs an uninitialized
`ViTModel` from this config, constructs the adaptor and OMR decoder locally, and
then performs a strict safetensors load. Any missing or unexpected key aborts the
export.

The resulting standalone config is written into the ONNX bundle so the export is
reproducible without relying on a user-specific Hugging Face cache.

### Graph boundary

The graph is split after all page-dependent visual preparation:

```text
RGB page
  -> ViT encoder
  -> remove CLS token
  -> 1x1 adaptor
  -> 2D positional encoding
  -> raw_features + enhanced_features
  -> repeated full-prefix decoder calls
  -> greedy token ids
```

This boundary prevents the ViT, adaptor, and 2D positional encoding from being
recomputed for each generated token. It also avoids sending the larger
`[1, 4097, 768]` raw ViT output between sessions.

## ONNX Contracts

### `encoder.onnx`

Input:

| Name | Dtype | Shape | Contract |
| --- | --- | --- | --- |
| `pixel_values` | float32 | `[1, 3, 1024, 1024]` | RGB values in `[0, 1]` |

Outputs:

| Name | Dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `raw_features` | float32 | `[1, 4096, 256]` | Adapted encoder features |
| `enhanced_features` | float32 | `[1, 4096, 256]` | Raw features plus 2D position encoding |

The public ONNX layout is batch-first. The wrapper performs the sequence-first
permutations required by the PyTorch decoder internally.

### `decoder.onnx`

Inputs:

| Name | Dtype | Shape | Contract |
| --- | --- | --- | --- |
| `raw_features` | float32 | `[1, 4096, 256]` | Output of `encoder.onnx` |
| `enhanced_features` | float32 | `[1, 4096, 256]` | Output of `encoder.onnx` |
| `token_ids` | int64 | `[1, T]` | Complete prefix, `1 <= T < 7512` |

Output:

| Name | Dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `next_token_logits` | float32 | `[1, 215]` | Logits for the final prefix position |

Only the token axis `T` is dynamic. Batch size and page resolution remain fixed
at the audited production values. Internally the decoder computes every prefix
position through every layer and projects only the final hidden position, which
preserves uncached production semantics while avoiding an unnecessary full-logit
transfer back to the host.

### Greedy host loop

The ONNX Runtime helper:

1. preprocesses and encodes one page;
2. starts the prefix with BOS token id `100`;
3. calls `decoder.onnx` with the complete accumulated prefix;
4. appends `argmax(next_token_logits)`;
5. stops at EOS token id `183` or the configured maximum length `7512`;
6. converts generated ids through the bundled `i2w` vocabulary.

The host loop owns EOS handling and string conversion. These control-flow and
dictionary operations do not belong in either numeric graph.

## Preprocessing

Preprocessing must match `convert_img_to_tensor()`:

1. accept an image-like array or Pillow image;
2. convert to RGB;
3. resize directly to 1024 x 1024 using the existing torchvision/Pillow resize
   semantics;
4. convert to channel-first float32 in `[0, 1]`;
5. add a batch dimension.

There is no mean/std normalization. The bundle records this contract in
`preprocessor_config.json` and `metadata.json`.

## Files And Ownership

Implementation code is scoped to the full-page OMR experiment:

```text
experiments/full_page_omr/export_onnx.py
experiments/full_page_omr/onnx_runtime.py
tests/test_full_page_omr_onnx.py
```

Generated deployment assets are written to:

```text
experiments/full_page_omr/weights/polish_scores_cl_CL_onnx/
  encoder.onnx
  decoder.onnx
  config.json
  preprocessor_config.json
  metadata.json
```

The exporter defaults to the first checkpoint's safetensors/config pair but
accepts explicit paths. It builds and verifies all five files in a sibling
staging directory, then swaps the complete directory into place. A failed
re-export leaves the previous bundle intact. The runtime requires the completion
manifest and verifies the declared size and SHA-256 of both graphs and both
configuration files before creating sessions, so a mixed or modified bundle is
rejected.

## Export And Runtime Environment

- Exporter: PyTorch 2.13 ONNX exporter, legacy tracing path (`dynamo=False`),
  opset 20.
- Graph validation: `onnx.checker`.
- Runtime validation: the qinglong-captions `musvit-onnx` dependency profile,
  whose `onnx-base` profile selects ONNX Runtime GPU from the CUDA 13 package
  index on Windows.
- Execution providers: the CUDA validation gate explicitly requires the CUDA
  provider; it cannot silently degrade to CPU. A separate CPU-only session runs
  the encoder and one decoder step for portable smoke validation.

PyTorch performs ONNX serialization. ONNX Runtime is the deployment engine and
the independent execution oracle; it is not described as the exporter.

## Validation

### Structural gates

- Offline model construction does not invoke or read foundation weights.
- Strict state loading reports zero missing and zero unexpected keys.
- Both ONNX graphs pass `onnx.checker`.
- ONNX Runtime loads both sessions and exposes exactly the documented names,
  dtypes, and shapes.
- Neither graph uses external tensor-data sidecars.
- Each graph and their combined size remain below 2 GiB.

### Numerical gates

Using deterministic FP32 inputs:

- compare PyTorch and ONNX Runtime `raw_features`;
- compare PyTorch and ONNX Runtime `enhanced_features`;
- compare next-token logits for several prefix lengths, including length 1 and a
  multi-token prefix;
- require `rtol=1e-4` and `atol=1e-4` for every tensor comparison;
- require matching argmax ids for every checked prefix and record maximum
  absolute and relative errors in metadata or the verification report.

A tolerance failure blocks completion. The implementation must not silently
loosen these fixed limits in response to a failed export.

### End-to-end gate

Run PyTorch and ONNX Runtime greedy decoding on Polish Scores validation row 0 at
the locked dataset revision
`b3170c8b8f322885b566efe9e264af9328b5603f`. The complete generated token-id
sequence, EOS termination state, and decoded token strings must match. A mismatch
blocks completion even if tensor-level tolerances pass. The CUDA Execution
Provider is the primary end-to-end gate on the available CUDA 13 system; the CPU
Execution Provider must also load both graphs and pass a one-step smoke test.

## Error Handling

The exporter fails with an actionable error when:

- input safetensors or either config is absent;
- the embedded encoder config is incompatible with the state keys;
- strict loading fails;
- an input/output name or shape differs from the locked contract;
- ONNX checker or ONNX Runtime session creation fails;
- numerical or greedy parity fails;
- a final graph is empty, has sidecar tensor files, or exceeds 2 GiB.

Metadata includes source hashes, opset, framework/runtime versions, graph hashes,
tensor shapes, vocabulary ids, preprocessing contract, and validation results.

## Acceptance Criteria

The task is complete when the two ONNX graphs and companion JSON files exist in
the target bundle, pass all structural and numerical checks, reproduce one real
page's full PyTorch greedy token sequence, require no foundation weight download
at runtime, and remain below the 2 GiB size limit.
