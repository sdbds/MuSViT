# Windows Attention Dependencies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lock MuSViT's Windows Python 3.11 environment to CUDA 13 Torch 2.13 and add compatible Triton and FlashAttention packages.

**Architecture:** Platform markers keep the Windows ABI-specific packages separate from other platforms. uv continues to resolve Torch from the CUDA 13 index and resolves FlashAttention from the supplied prebuilt wheel URL.

**Tech Stack:** Python 3.11, uv, PyTorch 2.13, CUDA 13, triton-windows, flash-attn 2.8.4

## Global Constraints

- Add `triton-windows` only when `sys_platform == 'win32'`.
- Add `flash-attn==2.8.4` only when `sys_platform == 'win32' and python_version == '3.11'`.
- Use the supplied CPython 3.11 wheel compiled for CUDA 13 and Torch 2.13.0.
- Pin Windows to `torch==2.13.0` and `torchvision==0.28.0`.
- Do not synchronize or otherwise modify the active `.venv` while training is running.

---

### Task 1: Declare And Lock Windows Attention Dependencies

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: uv platform markers and the existing `pytorch-cu130` index.
- Produces: a reproducible Windows Python 3.11 resolution for Torch, Triton, and FlashAttention.

- [x] **Step 1: Split Torch constraints and add Windows-only packages**

Replace the unqualified Torch requirements with platform-specific requirements and add:

```toml
"torch>=2.6.0; sys_platform != 'win32'",
"torch==2.13.0; sys_platform == 'win32'",
"torchvision>=0.21.0; sys_platform != 'win32'",
"torchvision==0.28.0; sys_platform == 'win32'",
"triton-windows; sys_platform == 'win32'",
"flash-attn==2.8.4; sys_platform == 'win32' and python_version == '3.11'",
```

- [x] **Step 2: Add the FlashAttention uv source**

Add this source under `[tool.uv.sources]`:

```toml
flash-attn = [
    { url = "https://github.com/sdbds/flash-attention-for-windows/releases/download/2.8.4/flash_attn-2.8.4+cu130torch2.13.0cxx11abiFALSEfullbackward-cp311-cp311-win_amd64.whl", marker = "sys_platform == 'win32' and python_version == '3.11'" },
]
```

- [x] **Step 3: Regenerate the lock file without synchronizing the environment**

Run:

```powershell
uv lock
```

Expected: resolution succeeds and `uv.lock` gains `flash-attn` and `triton-windows` packages while retaining `torch==2.13.0+cu130` and `torchvision==0.28.0+cu130` for Windows.

- [x] **Step 4: Verify the frozen resolution**

Run:

```powershell
uv lock --check
rg -n 'name = "(flash-attn|triton-windows)"|version = "(2.13.0\+cu130|0.28.0\+cu130)"' uv.lock
```

Expected: `uv lock --check` exits successfully and all four Windows packages appear in `uv.lock`.

- [x] **Step 5: Review the scoped diff**

Run:

```powershell
git diff --check
git diff -- pyproject.toml uv.lock
```

Expected: no whitespace errors, no environment synchronization, and no changes outside dependency declarations and lock data.
