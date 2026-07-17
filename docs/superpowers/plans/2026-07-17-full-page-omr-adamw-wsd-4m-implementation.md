# Full-page OMR AdamW + WSD 4M Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the full-page OMR constant-Adam training path with an audited four-group AdamW optimizer and native Transformers WSD schedule for a fresh 4,000,000-step run.

**Architecture:** Add a focused `optimization.py` module that owns immutable optimizer/scheduler configuration, parameter partitioning, AdamW construction, WSD construction, and metadata. `SMTPP_Trainer` consumes that module and enforces checkpoint compatibility; `finetune.py`, `entrypoint.py`, and PowerShell only normalize and transport explicit protocol values.

**Tech Stack:** Python 3.11, PyTorch, Lightning, Transformers 4.57.5 `get_wsd_schedule`, Fire CLI, PowerShell, unittest/pytest.

## Global Constraints

- Work only in `D:\UGit\MuSViT\.worktrees\full-page-omr-eval-v2` on branch `codex/full-page-omr-eval-v2`; do not modify or merge into `main` while PID 26704 or its workers are alive.
- Run protocol is `full_page_omr_adamw_wsd_4m_v1`; metric protocol remains `canonical_v2` and checkpoint monitor remains `val_SER_v2`.
- `max_steps=4_000_000`, `batch_size=1`, `accumulate_grad_batches=1`, and `max_epochs=100_000`.
- AdamW task LR is `1e-4`, encoder LR is `1e-5`, ordinary weight decay is `0.01`, bias/LayerNorm decay is `0`, betas are `(0.9, 0.999)`, eps is `1e-8`, and amsgrad is false.
- WSD uses linear warmup for 10,000 steps, stable LR for 3,590,000 steps, cosine decay for 400,000 steps, `min_lr_ratio=0`, and step interval.
- Validation uses `check_val_every_n_epoch=2_000`, `val_check_interval=1.0`, and no sanity validation; the first validation is epoch 2,000.
- Curriculum remains unchanged: real data and encoder unfreeze at step 120,000; steady 20% synthetic / 80% real mixture begins at step 320,000.
- Preserve baseline Polish Scores `reduce_ratio=0.5`, `precision=16-mixed`, uncached greedy decoding, no gradient clipping, and no EarlyStopping.
- Every production change follows a red-green TDD cycle and each task ends in an independently reviewable commit.

---

## File Structure

- Create `experiments/full_page_omr/optimization.py`: immutable AdamW/WSD config, validation, four-group partitioning, optimizer/scheduler factories, and audit metadata.
- Create `tests/test_full_page_omr_optimizer.py`: focused unit tests for group ownership, decay exclusions, WSD boundaries, and scheduler state restoration.
- Modify `experiments/full_page_omr/smt_trainer.py`: consume the optimization module, return Lightning optimizer/scheduler config, and enforce resume identity.
- Modify `experiments/full_page_omr/finetune.py`: normalize CLI values, build protocol metadata, validate checkpoint optimizer identity, and configure epoch validation.
- Modify `experiments/full_page_omr/entrypoint.py`: transport canonical optimizer/WSD arguments and the deprecated `learning_rate` alias.
- Modify `experiments/full_page_omr/_globals.py`: remove the optimizer learning-rate global; retain only resolution.
- Modify `2.full_page_omr.ps1`: lock the 4M AdamW/WSD protocol and 2,000-epoch validation cadence.
- Modify `tests/test_full_page_omr_throughput.py`: update launch, Trainer, metadata, resume, and PowerShell contracts.

---

### Task 1: AdamW Configuration And Parameter Ownership

**Files:**
- Create: `experiments/full_page_omr/optimization.py`
- Create: `tests/test_full_page_omr_optimizer.py`

**Interfaces:**
- Produces: `AdamWWSDConfig`, `build_adamw_parameter_groups(model, config)`, and `build_adamw(model, config)`.
- Consumes: a model with an `encoder` submodule; no Lightning dependency.

- [ ] **Step 1: Write failing configuration and partition tests**

Create `tests/test_full_page_omr_optimizer.py`:

```python
import unittest

import torch
from torch import nn

from experiments.full_page_omr.optimization import (
    AdamWWSDConfig,
    build_adamw,
    build_adamw_parameter_groups,
)


class TinyOMRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        self.adaptor = nn.Conv2d(4, 4, kernel_size=1)
        self.decoder = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        self.in_proj_bias = nn.Parameter(torch.zeros(4))
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False


class AdamWWSDParameterGroupTests(unittest.TestCase):
    def setUp(self):
        self.model = TinyOMRModel()
        self.config = AdamWWSDConfig()

    def test_default_config_matches_locked_protocol(self):
        self.assertEqual(self.config.max_steps, 4_000_000)
        self.assertEqual(self.config.stable_steps, 3_590_000)
        self.assertEqual(self.config.task_learning_rate, 1e-4)
        self.assertEqual(self.config.encoder_learning_rate, 1e-5)

    def test_each_parameter_appears_once_and_frozen_encoder_is_included(self):
        groups = build_adamw_parameter_groups(self.model, self.config)
        grouped = [parameter for group in groups for parameter in group["params"]]
        self.assertEqual(len(grouped), len(list(self.model.parameters())))
        self.assertEqual(len({id(parameter) for parameter in grouped}), len(grouped))
        encoder_ids = {id(parameter) for parameter in self.model.encoder.parameters()}
        self.assertTrue(encoder_ids.issubset({id(parameter) for parameter in grouped}))

    def test_bias_and_layer_norm_are_no_decay(self):
        groups = build_adamw_parameter_groups(self.model, self.config)
        by_id = {
            id(parameter): group["weight_decay"]
            for group in groups
            for parameter in group["params"]
        }
        for module in self.model.modules():
            for parameter_name, parameter in module.named_parameters(recurse=False):
                expected_no_decay = "bias" in parameter_name.lower() or isinstance(module, nn.LayerNorm)
                self.assertEqual(by_id[id(parameter)] == 0.0, expected_no_decay)

    def test_adamw_group_lrs_and_decay_match_protocol(self):
        optimizer = build_adamw(self.model, self.config)
        actual = {
            group["name"]: (group["lr"], group["weight_decay"])
            for group in optimizer.param_groups
        }
        self.assertEqual(actual["encoder_decay"], (1e-5, 0.01))
        self.assertEqual(actual["encoder_no_decay"], (1e-5, 0.0))
        self.assertEqual(actual["task_decay"], (1e-4, 0.01))
        self.assertEqual(actual["task_no_decay"], (1e-4, 0.0))
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.defaults["betas"], (0.9, 0.999))
        self.assertEqual(optimizer.defaults["eps"], 1e-8)
        self.assertFalse(optimizer.defaults["amsgrad"])

    def test_encoder_parameter_updates_after_it_is_unfrozen(self):
        optimizer = build_adamw(self.model, self.config)
        parameter = next(self.model.encoder.parameters())
        before = parameter.detach().clone()
        parameter.requires_grad = True
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        self.assertFalse(torch.equal(parameter, before))
```

- [ ] **Step 2: Run the focused tests and verify RED**

```powershell
$env:CAIROCFFI_DLL_DIRECTORIES='D:\BaiduNetdisk\module\ImageViewer'
$env:PATH="$env:CAIROCFFI_DLL_DIRECTORIES;$env:PATH"
.\.venv\Scripts\python.exe -m pytest tests\test_full_page_omr_optimizer.py -q
```

Expected: collection fails with `ModuleNotFoundError: experiments.full_page_omr.optimization`.

- [ ] **Step 3: Implement immutable config validation and four-group partitioning**

Create `experiments/full_page_omr/optimization.py`:

```python
from dataclasses import asdict, dataclass

import torch
from torch import nn
from transformers.optimization import get_wsd_schedule


OPTIMIZER_PROTOCOL = "adamw_wsd_v1"


@dataclass(frozen=True)
class AdamWWSDConfig:
    protocol: str = OPTIMIZER_PROTOCOL
    task_learning_rate: float = 1e-4
    encoder_learning_rate: float = 1e-5
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False
    max_steps: int = 4_000_000
    warmup_steps: int = 10_000
    decay_steps: int = 400_000
    warmup_type: str = "linear"
    decay_type: str = "cosine"
    min_lr_ratio: float = 0.0
    num_cycles: float = 0.5

    def __post_init__(self):
        for name in ("task_learning_rate", "encoder_learning_rate", "eps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if (isinstance(self.weight_decay, bool)
                or not isinstance(self.weight_decay, (int, float))
                or self.weight_decay < 0):
            raise ValueError("weight_decay must be non-negative")
        if len(self.betas) != 2 or not all(0 <= beta < 1 for beta in self.betas):
            raise ValueError("betas must contain two values in [0, 1)")
        for name in ("max_steps", "warmup_steps", "decay_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.max_steps <= 0 or self.warmup_steps + self.decay_steps >= self.max_steps:
            raise ValueError("WSD phases must leave a positive stable interval")
        if self.warmup_type not in {"linear", "cosine", "1-sqrt"}:
            raise ValueError("unsupported WSD warmup_type")
        if self.decay_type not in {"linear", "cosine", "1-sqrt"}:
            raise ValueError("unsupported WSD decay_type")
        if not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be between 0 and 1")

    @property
    def stable_steps(self) -> int:
        return self.max_steps - self.warmup_steps - self.decay_steps

    def to_metadata(self) -> dict:
        metadata = asdict(self)
        metadata["betas"] = list(self.betas)
        metadata["stable_steps"] = self.stable_steps
        return metadata


def _no_decay_parameter_ids(model: nn.Module) -> set[int]:
    result = set()
    for module in model.modules():
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if "bias" in parameter_name.lower() or isinstance(module, nn.LayerNorm):
                result.add(id(parameter))
    return result


def build_adamw_parameter_groups(model: nn.Module, config: AdamWWSDConfig) -> list[dict]:
    if not hasattr(model, "encoder"):
        raise ValueError("full-page OMR model must expose an encoder submodule")
    encoder_ids = {id(parameter) for parameter in model.encoder.parameters()}
    no_decay_ids = _no_decay_parameter_ids(model)
    buckets = {name: [] for name in (
        "encoder_decay", "encoder_no_decay", "task_decay", "task_no_decay"
    )}
    for parameter in model.parameters():
        owner = "encoder" if id(parameter) in encoder_ids else "task"
        decay = "no_decay" if id(parameter) in no_decay_ids else "decay"
        buckets[f"{owner}_{decay}"].append(parameter)
    empty = [name for name, parameters in buckets.items() if not parameters]
    if empty:
        raise ValueError(f"empty AdamW parameter groups: {empty}")
    groups = []
    for name, parameters in buckets.items():
        groups.append({
            "name": name,
            "params": parameters,
            "lr": config.encoder_learning_rate if name.startswith("encoder_") else config.task_learning_rate,
            "weight_decay": 0.0 if name.endswith("_no_decay") else config.weight_decay,
        })
    grouped_ids = [id(parameter) for group in groups for parameter in group["params"]]
    model_ids = [id(parameter) for parameter in model.parameters()]
    if len(grouped_ids) != len(set(grouped_ids)) or set(grouped_ids) != set(model_ids):
        raise ValueError("AdamW parameter groups must partition model parameters exactly once")
    return groups


def build_adamw(model: nn.Module, config: AdamWWSDConfig) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        build_adamw_parameter_groups(model, config),
        betas=config.betas,
        eps=config.eps,
        amsgrad=config.amsgrad,
    )
```

- [ ] **Step 4: Run parameter-group tests and verify GREEN**

Run the Step 2 command.

Expected: all tests in `test_full_page_omr_optimizer.py` pass.

- [ ] **Step 5: Commit the optimizer core**

```powershell
git add experiments/full_page_omr/optimization.py tests/test_full_page_omr_optimizer.py
git commit -m "feat: add AdamW parameter groups for OMR"
```

---

### Task 2: Native WSD And Lightning Integration

**Files:**
- Modify: `experiments/full_page_omr/optimization.py`
- Modify: `experiments/full_page_omr/smt_trainer.py`
- Modify: `tests/test_full_page_omr_optimizer.py`
- Modify: `tests/test_full_page_omr_throughput.py`

**Interfaces:**
- Consumes: `AdamWWSDConfig` and factories from Task 1.
- Produces: `build_wsd_scheduler`, `optimizer_protocol_metadata`, and one Lightning AdamW/WSD optimizer configuration.

- [ ] **Step 1: Add failing WSD boundary and restore tests**

Append to `tests/test_full_page_omr_optimizer.py`:

```python
from experiments.full_page_omr.optimization import build_wsd_scheduler


class AdamWWSDTransitionTests(unittest.TestCase):
    def setUp(self):
        self.model = TinyOMRModel()
        self.config = AdamWWSDConfig()

    def test_wsd_lambda_boundaries(self):
        optimizer = build_adamw(self.model, self.config)
        scheduler = build_wsd_scheduler(optimizer, self.config)
        lr_lambda = scheduler.lr_lambdas[0]
        self.assertEqual(lr_lambda(0), 0.0)
        self.assertEqual(lr_lambda(10_000), 1.0)
        self.assertEqual(lr_lambda(3_600_000), 1.0)
        self.assertEqual(lr_lambda(4_000_000), 0.0)

    def test_optimizer_and_scheduler_state_resume_without_lr_jump(self):
        optimizer = build_adamw(self.model, self.config)
        scheduler = build_wsd_scheduler(optimizer, self.config)
        for _ in range(25):
            optimizer.step()
            scheduler.step()
        optimizer_state = optimizer.state_dict()
        scheduler_state = scheduler.state_dict()
        expected_lr = scheduler.get_last_lr()

        restored_model = TinyOMRModel()
        restored_optimizer = build_adamw(restored_model, self.config)
        restored_scheduler = build_wsd_scheduler(restored_optimizer, self.config)
        restored_optimizer.load_state_dict(optimizer_state)
        restored_scheduler.load_state_dict(scheduler_state)
        self.assertEqual(restored_scheduler.get_last_lr(), expected_lr)

        optimizer.step()
        scheduler.step()
        restored_optimizer.step()
        restored_scheduler.step()
        self.assertEqual(restored_scheduler.get_last_lr(), scheduler.get_last_lr())
```

In `tests/test_full_page_omr_throughput.py`, replace the existing bare-Adam assertion with:

```python
configured = module.configure_optimizers()
self.assertIsInstance(configured["optimizer"], torch.optim.AdamW)
self.assertEqual(configured["lr_scheduler"]["interval"], "step")
self.assertEqual(configured["lr_scheduler"]["frequency"], 1)
```

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
$env:CAIROCFFI_DLL_DIRECTORIES='D:\BaiduNetdisk\module\ImageViewer'
$env:PATH="$env:CAIROCFFI_DLL_DIRECTORIES;$env:PATH"
.\.venv\Scripts\python.exe -m pytest tests\test_full_page_omr_optimizer.py tests\test_full_page_omr_throughput.py -q
```

Expected: failures because `build_wsd_scheduler` does not exist and the trainer still returns `torch.optim.Adam`.

- [ ] **Step 3: Implement native WSD and metadata factories**

Append to `optimization.py`:

```python
def build_wsd_scheduler(optimizer, config: AdamWWSDConfig):
    return get_wsd_schedule(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_decay_steps=config.decay_steps,
        num_training_steps=config.max_steps,
        warmup_type=config.warmup_type,
        decay_type=config.decay_type,
        min_lr_ratio=config.min_lr_ratio,
        num_cycles=config.num_cycles,
    )


def optimizer_protocol_metadata(model: nn.Module, config: AdamWWSDConfig) -> dict:
    groups = build_adamw_parameter_groups(model, config)
    return {
        "optimizer": "AdamW",
        "optimizer_protocol": config.protocol,
        "optimizer_betas": list(config.betas),
        "optimizer_eps": config.eps,
        "optimizer_amsgrad": config.amsgrad,
        "optimizer_groups": [
            {
                "name": group["name"],
                "parameter_count": sum(parameter.numel() for parameter in group["params"]),
                "learning_rate": group["lr"],
                "weight_decay": group["weight_decay"],
            }
            for group in groups
        ],
        "scheduler": "transformers.get_wsd_schedule",
        "scheduler_interval": "step",
        "scheduler_frequency": 1,
        "wsd_max_steps": config.max_steps,
        "wsd_warmup_steps": config.warmup_steps,
        "wsd_stable_steps": config.stable_steps,
        "wsd_decay_steps": config.decay_steps,
        "wsd_warmup_type": config.warmup_type,
        "wsd_decay_type": config.decay_type,
        "wsd_min_lr_ratio": config.min_lr_ratio,
        "wsd_num_cycles": config.num_cycles,
        "task_learning_rate": config.task_learning_rate,
        "encoder_learning_rate": config.encoder_learning_rate,
        "weight_decay": config.weight_decay,
    }
```

- [ ] **Step 4: Replace the trainer's global Adam with explicit AdamW/WSD**

In `smt_trainer.py`, remove `from . import _globals`, import Task 2 APIs, add these constructor arguments, and create `self.optimizer_config` before the single existing `save_hyperparameters` call:

```python
run_protocol_version="full_page_omr_adamw_wsd_4m_v1",
optimizer_protocol="adamw_wsd_v1",
task_learning_rate=1e-4,
encoder_learning_rate=1e-5,
weight_decay=0.01,
max_steps=4_000_000,
wsd_warmup_steps=10_000,
wsd_decay_steps=400_000,
wsd_warmup_type="linear",
wsd_decay_type="cosine",
wsd_min_lr_ratio=0.0,
```

```python
self.optimizer_config = AdamWWSDConfig(
    protocol=optimizer_protocol,
    task_learning_rate=task_learning_rate,
    encoder_learning_rate=encoder_learning_rate,
    weight_decay=weight_decay,
    max_steps=max_steps,
    warmup_steps=wsd_warmup_steps,
    decay_steps=wsd_decay_steps,
    warmup_type=wsd_warmup_type,
    decay_type=wsd_decay_type,
    min_lr_ratio=wsd_min_lr_ratio,
)


def configure_optimizers(self):
    optimizer = build_adamw(self.model, self.optimizer_config)
    scheduler = build_wsd_scheduler(optimizer, self.optimizer_config)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "name": "wsd",
        },
    }


def optimizer_protocol_metadata(self):
    return optimizer_protocol_metadata(self.model, self.optimizer_config)
```

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the Step 2 command.

Expected: optimizer, WSD, and existing trainer tests pass.

- [ ] **Step 6: Commit Lightning integration**

```powershell
git add experiments/full_page_omr/optimization.py experiments/full_page_omr/smt_trainer.py tests/test_full_page_omr_optimizer.py tests/test_full_page_omr_throughput.py
git commit -m "feat: train OMR with AdamW and WSD"
```

---

### Task 3: Checkpoint Optimizer Protocol Enforcement

**Files:**
- Modify: `experiments/full_page_omr/smt_trainer.py`
- Modify: `experiments/full_page_omr/finetune.py`
- Modify: `tests/test_full_page_omr_optimizer.py`
- Modify: `tests/test_full_page_omr_throughput.py`

**Interfaces:**
- Consumes: optimizer hyperparameters saved by Task 2.
- Produces: full resume only for an identical AdamW/WSD protocol; weights-only loading bypasses optimizer-state compatibility.

- [ ] **Step 1: Add failing resume identity tests**

In the existing `SMTPPTrainerThroughputTests`, construct `SMTPP_Trainer(SimpleNamespace(padding_token=0), _TinyModel(), encoder_unfreeze_step=120000)` and add:

```python
def test_full_resume_rejects_legacy_adam_checkpoint(self):
    module = SMTPP_Trainer(
        SimpleNamespace(padding_token=0),
        _TinyModel(),
        encoder_unfreeze_step=120000,
    )
    checkpoint = {"hyper_parameters": {
        "encoder_training_mode": "fine_tune",
        "encoder_unfreeze_step": 120000,
        "curriculum_step_offset": 0,
    }}
    with self.assertRaisesRegex(ValueError, "optimizer protocol"):
        module.on_load_checkpoint(checkpoint)

def test_weights_only_load_may_bypass_legacy_optimizer_protocol(self):
    module = SMTPP_Trainer(
        SimpleNamespace(padding_token=0),
        _TinyModel(),
        encoder_unfreeze_step=120000,
        enforce_checkpoint_protocol=False,
    )
    module.on_load_checkpoint({"hyper_parameters": {}})

def test_full_resume_accepts_exact_adamw_wsd_identity(self):
    module = SMTPP_Trainer(
        SimpleNamespace(padding_token=0),
        _TinyModel(),
        encoder_unfreeze_step=120000,
    )
    configured = module.configure_optimizers()
    checkpoint = {
        "hyper_parameters": dict(module.hparams),
        "optimizer_states": [configured["optimizer"].state_dict()],
        "lr_schedulers": [configured["lr_scheduler"]["scheduler"].state_dict()],
    }
    module.on_load_checkpoint(checkpoint)
```

Extend `_validate_run_contract` tests with a real temporary checkpoint whose `hyper_parameters` omit `optimizer_protocol`; full resume must raise before `Trainer.fit`.

- [ ] **Step 2: Run resume tests and verify RED**

```powershell
$env:CAIROCFFI_DLL_DIRECTORIES='D:\BaiduNetdisk\module\ImageViewer'
$env:PATH="$env:CAIROCFFI_DLL_DIRECTORIES;$env:PATH"
.\.venv\Scripts\python.exe -m pytest tests\test_full_page_omr_optimizer.py tests\test_full_page_omr_throughput.py -q
```

Expected: legacy checkpoint tests do not raise and checkpoint run state lacks optimizer identity.

- [ ] **Step 3: Enforce exact optimizer hyperparameters in `SMTPP_Trainer`**

Add and invoke these methods near the start of `on_load_checkpoint`, after the existing `enforce_checkpoint_protocol` early return:

```python
def _optimizer_checkpoint_identity(self):
    config = self.optimizer_config
    return {
        "run_protocol_version": self.hparams.run_protocol_version,
        "optimizer_protocol": config.protocol,
        "task_learning_rate": config.task_learning_rate,
        "encoder_learning_rate": config.encoder_learning_rate,
        "weight_decay": config.weight_decay,
        "max_steps": config.max_steps,
        "wsd_warmup_steps": config.warmup_steps,
        "wsd_decay_steps": config.decay_steps,
        "wsd_warmup_type": config.warmup_type,
        "wsd_decay_type": config.decay_type,
        "wsd_min_lr_ratio": config.min_lr_ratio,
    }

def _validate_optimizer_checkpoint_identity(self, hyper_parameters):
    expected = self._optimizer_checkpoint_identity()
    missing = [key for key in expected if key not in hyper_parameters]
    if missing:
        raise ValueError(f"checkpoint is missing optimizer protocol evidence: {missing}")
    mismatches = {
        key: (hyper_parameters[key], value)
        for key, value in expected.items()
        if hyper_parameters[key] != value
    }
    if mismatches:
        raise ValueError(f"checkpoint optimizer protocol mismatch: {mismatches}")
```

Also require exactly one optimizer state and one scheduler state. Compare each saved optimizer group's `name`, `initial_lr`, and `weight_decay` against the current group schema before Lightning can overwrite it:

```python
def _validate_optimizer_checkpoint_state(self, checkpoint):
    optimizer_states = checkpoint.get("optimizer_states")
    scheduler_states = checkpoint.get("lr_schedulers")
    if not isinstance(optimizer_states, list) or len(optimizer_states) != 1:
        raise ValueError("full resume requires exactly one AdamW optimizer state")
    if not isinstance(scheduler_states, list) or len(scheduler_states) != 1:
        raise ValueError("full resume requires exactly one WSD scheduler state")
    saved_groups = optimizer_states[0].get("param_groups", [])
    expected_groups = self.optimizer_protocol_metadata()["optimizer_groups"]
    saved_schema = [
        (group.get("name"), group.get("initial_lr"), group.get("weight_decay"))
        for group in saved_groups
    ]
    expected_schema = [
        (group["name"], group["learning_rate"], group["weight_decay"])
        for group in expected_groups
    ]
    if saved_schema != expected_schema:
        raise ValueError(
            f"checkpoint AdamW parameter-group mismatch: "
            f"saved={saved_schema}, expected={expected_schema}"
        )
```

Reject missing groups, reordered names, wrong base LR, wrong decay, weights-only checkpoints, and legacy Adam states.

- [ ] **Step 4: Carry optimizer identity in `CheckpointRunState`**

In `finetune.py`, add `optimizer_identity: dict | None` to `CheckpointRunState`. `_read_checkpoint_run_state()` reads `run_protocol_version` plus the ten optimizer/WSD keys returned by `_optimizer_checkpoint_identity()` from `hyper_parameters`. `_validate_run_contract(..., expected_optimizer_identity=...)` compares them only in the `from_checkpoint` branch. The `starting_weights` branch continues to validate source SHA/curriculum but deliberately ignores the source optimizer.

Require resumed `max_steps` to equal the checkpoint WSD total. A longer continuation needs a new protocol rather than extending this cosine curve.

- [ ] **Step 5: Run resume tests and verify GREEN**

Run the Step 2 command.

Expected: exact new checkpoints pass, legacy/full mismatches fail, and weights-only loading still passes.

- [ ] **Step 6: Commit checkpoint enforcement**

```powershell
git add experiments/full_page_omr/smt_trainer.py experiments/full_page_omr/finetune.py tests/test_full_page_omr_optimizer.py tests/test_full_page_omr_throughput.py
git commit -m "fix: enforce OMR optimizer resume identity"
```

---

### Task 4: CLI, Trainer Cadence, And Audit Metadata

**Files:**
- Modify: `experiments/full_page_omr/entrypoint.py`
- Modify: `experiments/full_page_omr/finetune.py`
- Modify: `experiments/full_page_omr/_globals.py`
- Modify: `tests/test_full_page_omr_throughput.py`

**Interfaces:**
- Consumes: `SMTPP_Trainer` constructor and metadata method from Tasks 2-3.
- Produces: canonical CLI arguments, deprecated single-LR alias normalization, 2,000-epoch validation, and complete local/W&B metadata.

- [ ] **Step 1: Add failing launch and Trainer contract tests**

Update entrypoint/launch tests to expect:

```python
self.assertEqual(signature.parameters["max_steps"].default, 4_000_000)
self.assertEqual(signature.parameters["validation_every_n_epochs"].default, 2_000)
self.assertEqual(signature.parameters["task_learning_rate"].default, None)
self.assertEqual(signature.parameters["encoder_learning_rate"].default, 1e-5)
self.assertEqual(signature.parameters["weight_decay"].default, 0.01)
self.assertEqual(signature.parameters["wsd_warmup_steps"].default, 10_000)
self.assertEqual(signature.parameters["wsd_decay_steps"].default, 400_000)
```

Replace Trainer kwargs assertions with:

```python
kwargs = finetune._build_trainer_kwargs(
    max_steps=4_000_000,
    validation_every_n_epochs=2_000,
    callbacks=[],
    logger=False,
)
self.assertEqual(kwargs["check_val_every_n_epoch"], 2_000)
self.assertEqual(kwargs["val_check_interval"], 1.0)
self.assertEqual(kwargs["max_steps"], 4_000_000)
self.assertEqual(kwargs["num_sanity_val_steps"], 0)
```

Add alias normalization tests:

```python
self.assertEqual(finetune._normalize_task_learning_rate(None, None), 1e-4)
self.assertEqual(finetune._normalize_task_learning_rate(2e-4, None), 2e-4)
self.assertEqual(finetune._normalize_task_learning_rate(None, 3e-4), 3e-4)
with self.assertRaisesRegex(ValueError, "mutually exclusive"):
    finetune._normalize_task_learning_rate(2e-4, 3e-4)
```

- [ ] **Step 2: Run throughput tests and verify RED**

```powershell
$env:CAIROCFFI_DLL_DIRECTORIES='D:\BaiduNetdisk\module\ImageViewer'
$env:PATH="$env:CAIROCFFI_DLL_DIRECTORIES;$env:PATH"
.\.venv\Scripts\python.exe -m pytest tests\test_full_page_omr_throughput.py -q
```

Expected: old 320k/batch-validation defaults and missing optimizer arguments fail.

- [ ] **Step 3: Normalize and transport canonical arguments**

Set `finetune.PROTOCOL_VERSION = "full_page_omr_adamw_wsd_4m_v1"`. Replace `validation_every_n_batches` with `validation_every_n_epochs`. Add task/encoder LR, weight decay, and WSD arguments to `entrypoint.run`, `finetune.launch`, and `finetune.main`.

Implement:

```python
def _normalize_task_learning_rate(task_learning_rate, learning_rate):
    if task_learning_rate is not None and learning_rate is not None:
        raise ValueError("task_learning_rate and learning_rate are mutually exclusive")
    value = task_learning_rate if task_learning_rate is not None else learning_rate
    if value is None:
        value = 1e-4
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError("task_learning_rate must be positive")
    return float(value)
```

Pass `run_protocol_version=protocol_version` and all normalized optimizer primitives to both fresh `SMTPP_Trainer(...)` and weights-only `SMTPP_Trainer.load_from_checkpoint(...)`. Remove `_globals.learning_rate` and its assignment; `_globals.py` retains only `resolution`.

- [ ] **Step 4: Implement epoch validation kwargs**

Replace `_validate_validation_every_n_batches` with `_validate_validation_every_n_epochs`, then implement:

```python
def _build_trainer_kwargs(*, max_steps, validation_every_n_epochs, callbacks, logger):
    return {
        "max_epochs": 100_000,
        "max_steps": max_steps,
        "check_val_every_n_epoch": validation_every_n_epochs,
        "val_check_interval": 1.0,
        "num_sanity_val_steps": 0,
        "callbacks": callbacks,
        "logger": logger,
        "precision": PRECISION,
        "accumulate_grad_batches": ACCUMULATE_GRAD_BATCHES,
    }
```

- [ ] **Step 5: Replace flat Adam metadata with audited AdamW/WSD metadata**

After model-wrapper construction, merge `model_wrapper.optimizer_protocol_metadata()` into `_build_protocol_metadata`. Record these fields in addition to the existing source, resize, precision, batch, metric, and run-record data:

```python
{
    "protocol_version": "full_page_omr_adamw_wsd_4m_v1",
    "max_steps": 4_000_000,
    "validation_every_n_epochs": 2_000,
    "validation_first_epoch": 2_000,
    "validation_expected_count": 24,
    "expected_training_batches_per_epoch": len(data.train_dataset),
    "encoder_unfreeze_step": 120_000,
    "curriculum_steady_mixture_step": 320_000,
    **model_wrapper.optimizer_protocol_metadata(),
}
```

Do not log the deprecated alias.

- [ ] **Step 6: Run throughput and optimizer tests and verify GREEN**

```powershell
$env:CAIROCFFI_DLL_DIRECTORIES='D:\BaiduNetdisk\module\ImageViewer'
$env:PATH="$env:CAIROCFFI_DLL_DIRECTORIES;$env:PATH"
.\.venv\Scripts\python.exe -m pytest tests\test_full_page_omr_throughput.py tests\test_full_page_omr_optimizer.py -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit CLI and audit integration**

```powershell
git add experiments/full_page_omr/entrypoint.py experiments/full_page_omr/finetune.py experiments/full_page_omr/_globals.py tests/test_full_page_omr_throughput.py
git commit -m "feat: expose audited 4M OMR training protocol"
```

---

### Task 5: Production PowerShell Contract

**Files:**
- Modify: `2.full_page_omr.ps1`
- Modify: `tests/test_full_page_omr_throughput.py`

**Interfaces:**
- Consumes: canonical CLI fields from Task 4.
- Produces: a dry-run-verifiable fresh 4M AdamW/WSD launch; Cairo remains environment-discovered with no machine-specific repository path.

- [ ] **Step 1: Update the PowerShell dry-run test first**

Require:

```python
self.assertIn("--max_steps=4000000", result.stdout)
self.assertIn("--validation_every_n_epochs=2000", result.stdout)
self.assertIn("--task_learning_rate=0.0001", result.stdout)
self.assertIn("--encoder_learning_rate=0.00001", result.stdout)
self.assertIn("--weight_decay=0.01", result.stdout)
self.assertIn("--wsd_warmup_steps=10000", result.stdout)
self.assertIn("--wsd_decay_steps=400000", result.stdout)
self.assertIn("--protocol_version=full_page_omr_adamw_wsd_4m_v1", result.stdout)
self.assertNotIn("--validation_every_n_batches", result.stdout)
```

- [ ] **Step 2: Run the PowerShell test and verify RED**

```powershell
$env:CAIROCFFI_DLL_DIRECTORIES='D:\BaiduNetdisk\module\ImageViewer'
$env:PATH="$env:CAIROCFFI_DLL_DIRECTORIES;$env:PATH"
.\.venv\Scripts\python.exe -m pytest tests\test_full_page_omr_throughput.py::FullPageOMRCheckpointTests::test_powershell_dry_run_includes_default_checkpoint_interval -q
```

Expected: output still contains 320000, batch validation, and the old protocol.

- [ ] **Step 3: Lock PowerShell defaults and validation**

Update `$Config`:

```powershell
max_steps                  = 4000000
validation_every_n_epochs = 2000
task_learning_rate        = 0.0001
encoder_learning_rate     = 0.00001
weight_decay              = 0.01
wsd_warmup_steps          = 10000
wsd_decay_steps           = 400000
wsd_warmup_type           = "linear"
wsd_decay_type            = "cosine"
wsd_min_lr_ratio          = 0.0
protocol_version           = "full_page_omr_adamw_wsd_4m_v1"
```

Validate positive LR values, non-negative weight decay/min ratio, positive max/validation/decay steps, non-negative warmup, supported types, and `warmup + decay < max_steps`. Add only canonical CLI flags to `$UvArgs`. Retain `from_checkpoint=$null`, `starting_weights=$null`, Cairo `$null`, baseline config path, and `checkpoint_every_n_epochs=100`.

- [ ] **Step 4: Run the PowerShell test and verify GREEN**

Run the Step 2 command.

Expected: one test passes and dry-run output contains all locked values.

- [ ] **Step 5: Commit the production launcher**

```powershell
git add 2.full_page_omr.ps1 tests/test_full_page_omr_throughput.py
git commit -m "feat: launch 4M AdamW WSD OMR training"
```

---

### Task 6: End-To-End Regression And Final Review

**Files:**
- Verify all files changed by Tasks 1-5; only modify them in response to a reproduced regression.

**Interfaces:**
- Consumes: complete new protocol.
- Produces: clean, fully tested branch ready for independent review; does not launch the 4M production run.

- [ ] **Step 1: Run optimizer and throughput suites together**

```powershell
$env:CAIROCFFI_DLL_DIRECTORIES='D:\BaiduNetdisk\module\ImageViewer'
$env:PATH="$env:CAIROCFFI_DLL_DIRECTORIES;$env:PATH"
.\.venv\Scripts\python.exe -m pytest tests\test_full_page_omr_optimizer.py tests\test_full_page_omr_throughput.py -q
```

Expected: all selected tests pass with no failures.

- [ ] **Step 2: Run the complete repository suite**

```powershell
$env:CAIROCFFI_DLL_DIRECTORIES='D:\BaiduNetdisk\module\ImageViewer'
$env:PATH="$env:CAIROCFFI_DLL_DIRECTORIES;$env:PATH"
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: all tests and subtests pass. Record fresh counts and runtime rather than copying the previous 155/271 result.

- [ ] **Step 3: Verify source hygiene and protocol strings**

```powershell
git diff --check dd3dc25..HEAD
rg -n "torch\.optim\.Adam\(|validation_every_n_batches|_globals\.learning_rate" experiments/full_page_omr 2.full_page_omr.ps1
git status --short --branch
```

Expected: `git diff --check` is clean; prohibited production strings have no active matches; worktree is clean after task commits.

- [ ] **Step 4: Inspect the PowerShell dry run**

```powershell
$env:CAIROCFFI_DLL_DIRECTORIES='D:\BaiduNetdisk\module\ImageViewer'
$env:PATH="$env:CAIROCFFI_DLL_DIRECTORIES;$env:PATH"
pwsh -NoProfile -File .\2.full_page_omr.ps1 -DryRun
```

Expected: fresh run, 4M max steps, 2,000-epoch validation, task/encoder LRs, AdamW decay, WSD phases, new protocol, and no machine-specific Cairo path.

- [ ] **Step 5: Commit test-driven cleanup if regression fixes were necessary**

If Steps 1-4 reproduced a defect, stage the complete known implementation surface and commit it:

```powershell
git add 2.full_page_omr.ps1 experiments/full_page_omr/optimization.py experiments/full_page_omr/smt_trainer.py experiments/full_page_omr/finetune.py experiments/full_page_omr/entrypoint.py experiments/full_page_omr/_globals.py tests/test_full_page_omr_optimizer.py tests/test_full_page_omr_throughput.py
git commit -m "fix: complete AdamW WSD OMR protocol"
```

If no correction was required, do not create an empty commit.

- [ ] **Step 6: Request independent review**

Ask the reviewer to verify parameter ownership, WSD boundary values, scheduler resume continuity, legacy Adam rejection, PowerShell defaults, and the full test evidence against `docs/superpowers/specs/2026-07-17-full-page-omr-adamw-wsd-4m-design.md`.
