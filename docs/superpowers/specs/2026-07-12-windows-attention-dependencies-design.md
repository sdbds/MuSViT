# Windows Attention Dependencies

## Goal

Make the default Windows Python 3.11 environment install the CUDA 13 builds required for Triton and FlashAttention while leaving non-Windows dependency resolution unchanged.

## Dependency Design

- Add `triton-windows` only when `sys_platform == 'win32'`.
- Add `flash-attn==2.8.4` only for Windows CPython 3.11.
- Resolve `flash-attn` from the supplied CUDA 13, Torch 2.13.0, CPython 3.11 wheel URL through `[tool.uv.sources]`.
- Pin the Windows Torch stack to `torch==2.13.0` and `torchvision==0.28.0` because the FlashAttention wheel is compiled against that exact Torch ABI.
- Retain the existing lower bounds for Torch and TorchVision on other platforms.

## Alternatives Considered

1. Main dependencies with platform markers. Selected because MuSViT's Windows training path is the consumer and should work after a normal `uv sync`.
2. A separate optional extra. Rejected because it makes the documented default training environment incomplete.
3. A PEP 508 direct URL in `project.dependencies`. Rejected because a uv source override keeps platform-specific distribution details outside the published core metadata.

## Locking And Installation

Regenerate `uv.lock` after editing `pyproject.toml`. Do not synchronize the active `.venv` while the current training process is running. A later `uv sync --frozen` will install the locked Windows packages.

## Verification

- Run `uv lock --check` after regeneration.
- Inspect the lock entries and confirm Windows resolves Torch `2.13.0+cu130`, TorchVision `0.28.0+cu130`, `triton-windows`, and the supplied FlashAttention wheel.
- Confirm non-Windows Torch dependencies retain their existing constraints.
