# Full-page OMR AdamW + WSD 4M 训练协议

## 目的

在不改动当前 live run、数据 curriculum、图像预处理和指标语义的前提下，为 Polish Scores CL 建立一个全新训练协议：

- 使用 AdamW，并对普通权重施加 `weight_decay=0.01`；
- 使用 encoder/task 分组学习率；
- 使用 Transformers 原生 WSD scheduler；
- 将训练上限提高到 4,000,000 optimizer steps；
- 每 2,000 epochs 验证一次；
- 保持 canonical v2 指标和既有 checkpoint/run 审计契约。

该协议从 foundation encoder 和随机初始化的 adaptor/decoder 开始，不从旧 OMR checkpoint 恢复。它是新的 task fine-tuning run，不是从零进行视觉预训练。

## 已确认模型结构

训练模型包含三部分：

1. `encoder`：从 `carlospm12/LSMT-MAE-Base-1024-16` 加载的预训练 ViT-MAE；
2. `adaptor`：随机初始化的 `1x1 Conv2d`，将 encoder 特征映射到 256 维；
3. `decoder`：随机初始化的 8 层 autoregressive decoder。

CL curriculum 保持不变：

```text
step 0..119999       encoder frozen, train adaptor + decoder
step 120000          real data enters and encoder unfreezes
step 120000..320000  synthetic ratio decreases toward 20%
step >= 320000       stable 20% synthetic / 80% real mixture
```

扩大总训练预算不得同比例推迟 curriculum 或 encoder 解冻点。

## 协议身份

新协议名固定为：

```text
full_page_omr_adamw_wsd_4m_v1
```

指标协议保持：

```text
metric_version = canonical_v2
checkpoint_monitor = val_SER_v2
```

W&B、本地 `protocol.json`、Lightning hyperparameters 和 checkpoint 必须记录新协议名。任何 optimizer、LR、WSD 阶段长度、总步数或 validation cadence 的修改都需要新的 protocol version，不得继续使用上述名字。

## 1. AdamW 参数组

### 1.1 基础参数

```text
optimizer = AdamW
task_learning_rate = 1e-4
encoder_learning_rate = 1e-5
weight_decay = 0.01
betas = (0.9, 0.999)
eps = 1e-8
amsgrad = false
```

上述 AdamW 参数全部显式传入。AdamW 实现版本进入审计 metadata，避免环境升级后同名协议产生不同语义。

### 1.2 四个参数组

模型参数必须被划分为四组：

| Group | Parameter ownership | LR | Weight decay |
| --- | --- | ---: | ---: |
| `encoder_decay` | encoder 普通权重 | `1e-5` | `0.01` |
| `encoder_no_decay` | encoder bias 与 LayerNorm 参数 | `1e-5` | `0` |
| `task_decay` | adaptor/decoder 普通权重 | `1e-4` | `0.01` |
| `task_no_decay` | adaptor/decoder bias 与 LayerNorm 参数 | `1e-4` | `0` |

`task` 定义为 `SMTFoundationModelForCausalLM` 中所有不属于 `encoder` 的可训练模块。参数归属使用模块和参数身份，不使用模糊的字符串包含判断。no-decay 集合只包含：

- 名为 `bias` 的参数；
- `torch.nn.LayerNorm` 模块直接拥有的参数。

每个模型参数必须恰好出现在一个 optimizer group。重复、遗漏或空的预期 group 都应在 `configure_optimizers()` 时失败。

Encoder 在 optimizer 构建时仍处于 frozen 状态，但它的全部参数必须从 step 0 起进入 optimizer groups。不得用 `parameter.requires_grad` 过滤 optimizer 参数，否则 step 120000 解冻后 encoder 不会被更新。冻结期间 encoder 没有 gradient，因此 AdamW 不会为其创建有效 moment state，也不会施加 weight decay。

## 2. Transformers WSD

### 2.1 固定阶段

使用 `transformers.optimization.get_wsd_schedule`：

```text
num_training_steps = 4,000,000
num_warmup_steps = 10,000
num_stable_steps = 3,590,000
num_decay_steps = 400,000
warmup_type = linear
decay_type = cosine
min_lr_ratio = 0
num_cycles = 0.5
```

三个阶段必须满足：

```text
10,000 + 3,590,000 + 400,000 = 4,000,000
```

调用原生 helper 时传入 `num_training_steps`、`num_warmup_steps` 和 `num_decay_steps`，由 helper 推导 stable 长度；审计 metadata 仍显式记录推导后的 `num_stable_steps=3,590,000`。

### 2.2 LR 语义

WSD 对四个 parameter groups 使用同一倍率：

```text
step 0             multiplier = 0
step 10000         multiplier = 1
step 10000..3599999 multiplier = 1
step 3600000       cosine decay begins
step 4000000       multiplier = 0
```

Encoder 在 step 120000 解冻时已经处于 stable 阶段，因此直接使用其较低 base LR `1e-5`。本协议不实现第二次 encoder-specific warmup。

Lightning scheduler 配置固定为：

```text
interval = step
frequency = 1
```

这里的 step 是 optimizer step，不是 `samples_seen` 或 curriculum step。当前 batch size 和 accumulation 都为 1，因此三者数值同步；协议仍保留不同概念，后续启用 accumulation 时不得复用这一偶然相等关系。

## 3. 训练终点与验证

### 3.1 Trainer 终点

```text
max_steps = 4,000,000
max_epochs = 100,000
batch_size = 1
accumulate_grad_batches = 1
```

`max_steps` 是真正终点。`max_epochs` 只提供高于预计约 48,193 epochs 的上界，不参与 WSD 长度计算。

### 3.2 Epoch validation cadence

新协议使用 Lightning 原生 epoch cadence：

```text
check_val_every_n_epoch = 2,000
val_check_interval = 1.0
num_sanity_val_steps = 0
```

验证点为 epoch 2000、4000、6000，依此类推。不存在单独的 validation warmup 或 3500-epoch offset。

Polish Scores 当前约 83 training batches/epoch，因此：

```text
first validation ~= step 166,000
validation spacing ~= 166,000 steps
4M run ~= 48,193 epochs
scheduled validation count ~= 24
```

这些 step 数只用于审阅；调度权威是 epoch。恢复训练时使用 checkpoint 的绝对 epoch，不能从恢复点重新计 2,000 epochs。

Metric checkpoint 保持：

```text
monitor = val_SER_v2
mode = min
save_top_k = 2
save_weights_only = true
```

EarlyStopping 保持禁用。每 100 epochs 覆盖保存的 full recovery checkpoint 保持不变，不与 metric checkpoint 混用。

若 `max_steps` 在非验证 epoch 中间结束，不额外制造一次未声明的 validation。训练结束后的 test 继续选择 `val_SER_v2` 最优 checkpoint。

## 4. 启动与恢复契约

### 4.1 默认启动

生产 PowerShell 默认启动必须是：

```text
from_checkpoint = None
starting_weights = None
data.skip_steps = 0
protocol_version = full_page_omr_adamw_wsd_4m_v1
```

Encoder 从固定 foundation weights 加载；adaptor 和 decoder 使用新随机初始化。随机种子行为不在本规格中修改。

### 4.2 Full resume

新协议产生的 full checkpoint 保存 model、AdamW state、WSD state、global step、epoch 和 `samples_seen`。full resume 必须校验：

- protocol version 完全一致；
- optimizer 为 AdamW；
- 四个 group 名称、base LR 和 weight decay 完全一致；
- WSD total/warmup/stable/decay/type/min ratio 完全一致；
- 请求的 `max_steps` 等于 checkpoint 中的 WSD total steps；
- 既有 curriculum offset 与 `samples_seen` 契约继续成立。

旧 Adam/constant-LR checkpoint 缺少上述证据，禁止 full resume。不得尝试把 Adam state dict 静默加载进 AdamW。

Lightning 负责恢复 scheduler `last_epoch` 和 optimizer moments。应用代码不得根据 `global_step` 手工二次推进 scheduler。

### 4.3 Weights-only fork

既有 `starting_weights` 入口继续表示新 Trainer：

- 只加载模型权重；
- 新建 AdamW 和 WSD state；
- optimizer step 从 0 开始；
- 仍要求来源 curriculum step、checkpoint SHA-256 和独立 protocol version。

本次批准的生产 run 是 fresh run，不使用该入口。

## 5. 配置与审计

PowerShell、CLI、`finetune.launch()`、`finetune.main()` 和 `SMTPP_Trainer` 之间必须显式贯通：

```text
max_steps
validation_every_n_epochs
task_learning_rate
encoder_learning_rate
weight_decay
wsd_warmup_steps
wsd_decay_steps
wsd_warmup_type
wsd_decay_type
wsd_min_lr_ratio
protocol_version
```

旧 `learning_rate` 单值不能继续作为 optimizer 的隐藏全局状态。CLI 保留 `learning_rate` 作为 `task_learning_rate` 的 deprecated compatibility alias：两者都未给出时使用 `1e-4`，只给出一个时使用该值，同时给出两者时失败。生产 PowerShell 只传规范字段 `task_learning_rate`，metadata 也只记录规范化后的字段。

`protocol.json`、W&B config 和 Lightning hyperparameters 至少记录：

- optimizer 类与 PyTorch 默认展开后的 `betas`、`eps`、`amsgrad`；
- 四个 group 的名称、参数数量、base LR 与 weight decay；
- WSD helper 名称和所有阶段参数；
- `max_steps`、`max_epochs` 和 scheduler interval；
- validation epoch cadence 与预计 training batches per epoch；
- encoder unfreeze step 和 curriculum steady-mixture step；
- metric version、checkpoint monitor、precision、batch size 和 accumulation；
- 既有 checkpoint/source/GPU/run identity 字段。

## 6. 不在本次修改中的项目

以下保持现状：

- `precision=16-mixed`；
- gradient clipping 未启用；
- batch size 与 accumulation 均为 1；
- Polish Scores baseline `reduce_ratio=0.5`；
- teacher-forcing noise、label smoothing 和 augmentation；
- uncached greedy generation 默认路径；
- beam search 与 repetition penalty；
- curriculum 阶段和 encoder 解冻边界。

这些变量需要独立 protocol，不得借 AdamW/WSD 实现顺带修改。

## 7. 涉及文件

- `2.full_page_omr.ps1`
- `experiments/full_page_omr/entrypoint.py`
- `experiments/full_page_omr/finetune.py`
- `experiments/full_page_omr/smt_trainer.py`
- `experiments/full_page_omr/_globals.py`
- `tests/test_full_page_omr_throughput.py`
- 新增 `tests/test_full_page_omr_optimizer.py`

不得修改 live run 所在主工作区。所有实现继续位于 `codex/full-page-omr-eval-v2` 的隔离 worktree。

## 8. 验收标准

### 参数组

1. 实测 optimizer 类型为 `torch.optim.AdamW`。
2. 每个模型参数恰好属于一个 group。
3. frozen encoder 参数仍存在于 optimizer。
4. encoder/task base LR 分别为 `1e-5` 与 `1e-4`。
5. bias 和 LayerNorm 参数 decay 为 0，其余参数 decay 为 `0.01`。

### Scheduler

6. 使用 Transformers `get_wsd_schedule`。
7. scheduler interval 为 step，frequency 为 1。
8. 自动化测试校验 step 0、10000、3600000 和4000000 的 LR 边界。
9. warmup、stable、decay 总和严格等于 4,000,000。
10. Encoder step 120000 解冻后能够获得 optimizer update，且使用 `1e-5` stable LR。

### Validation 与 checkpoint

11. Trainer 配置为每 2,000 epochs 验证一次，首次验证点为 epoch 2000。
12. checkpoint monitor 保持 `val_SER_v2`、top 2、weights-only。
13. EarlyStopping 不得重新出现。
14. 新协议 full resume 恢复 optimizer 和 scheduler state 后，下一步 LR 与未中断训练一致。
15. 旧 Adam checkpoint full resume 明确失败；weights-only 入口保持可用。

### 配置与回归

16. PowerShell dry run 输出 4M、epoch validation、AdamW 分组 LR 和 WSD 参数。
17. 本地 protocol 与 W&B metadata 包含完整 optimizer/scheduler identity。
18. 现有 canonical metrics、generation、curriculum、resize 和诊断测试全部继续通过。
19. `git diff --check` 通过，隔离 worktree 无未提交修改。
