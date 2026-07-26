# Full-page OMR DataLoader Worker 生命周期设计

## 状态

设计已确认，待实施。本规格取代
`2026-07-14-full-page-omr-throughput-design.md` 中“所有 DataLoader 共用
`num_workers=24`”的运行时约定，不改变其训练协议、模型和优化器结论。

## 目标

保留训练数据生成的现有吞吐，同时消除低频验证和测试阶段的 Windows
多进程内存峰值：

- 训练使用 `train_num_workers=24` 和 `persistent_workers=True`。
- 验证使用 `val_num_workers=0` 和 `persistent_workers=False`。
- 测试使用 `test_num_workers=0` 和 `persistent_workers=False`。

成功标准不是降低稳态训练内存，而是验证或测试开始时不再额外启动 24 个
Python 进程，也不再因系统提交内存耗尽产生 `WinError 1455`。

## 根因

Windows DataLoader 使用 `spawn`。每个 worker 都会创建新的 Python
解释器，并重新导入 PyTorch、CUDA、OpenCV、Pandas 和 Hugging Face
Datasets 等依赖。

当前训练 DataLoader 的 24 个 worker 常驻；验证 DataLoader 虽然已经设置为
非持久 worker，但仍会在低频验证开始时临时启动另外 24 个进程。训练 worker
不会在此期间退出，因此验证边界会形成约 48 个 worker 的瞬时峰值。验证和
checkpoint 周期重合时，checkpoint 序列化会进一步放大主进程内存压力。

## 方案比较

1. **拆分 train/val/test worker 数量，验证和测试同步加载。采用。**
   保留训练吞吐，并从根源移除验证、测试的进程峰值。验证每 200 epochs
   才执行一次，同步加载的额外墙钟时间相对完整训练可忽略。
2. **把所有 DataLoader 一起降到 4 至 8 个 worker。不采用。**
   实现最简单，但会无条件牺牲已测得的训练吞吐，偏离本次目标。
3. **让验证 worker 也常驻。不采用。**
   可避免反复 spawn，却会让训练和验证的两组 worker 长期同时占用提交内存，
   比当前方案风险更高。

## 配置契约

`ExperimentConfigWrapper.Data` 使用三个明确字段：

```json
{
  "train_num_workers": 24,
  "val_num_workers": 0,
  "test_num_workers": 0
}
```

规则：

- 三个值必须是非布尔整数且大于等于 0。
- 生产配置和 Windows PowerShell 运行时配置显式写入 `24/0/0`。
- 为读取旧配置，`from_dict()` 接受旧字段 `num_workers`，并将其解释为
  `train_num_workers`；旧配置未声明验证和测试数量时，两者默认是 0。
- 同一配置不得同时提供 `num_workers` 和 `train_num_workers`，避免出现两个
  训练 worker 真相来源。
- `to_dict()` 只输出三个新字段，不再生成旧字段。

## DataLoader 生命周期

三个 DataModule 使用相同规则：

| 阶段 | worker 数来源 | persistent workers |
| --- | --- | --- |
| train | `train_num_workers` | worker 数大于 0 时为 `True` |
| val | `val_num_workers` | 始终为 `False` |
| test | `test_num_workers` | 始终为 `False` |

`_build_dataloader()` 继续作为唯一构造入口：

- `num_workers=0` 时不传入 `prefetch_factor`、`worker_init_fn` 或
  `persistent_workers=True`。
- 训练 worker 数大于 0 时继续使用 `prefetch_factor=1`、worker seed 和
  pinned memory。
- 验证、测试即使以后配置为正数，也只按需创建并在迭代结束后退出。
- 不创建自定义进程池，不在 train、val 和 test 之间共享 Dataset worker。

## Checkpoint 兼容性

worker 数是运行时资源配置；验证和测试 worker 数不改变模型、优化器或训练
数据课程，因此不应阻止完整恢复。

现有 checkpoint 的 `protocol_snapshot` 含有：

```json
{"data": {"num_workers": 24}}
```

为保持这些 checkpoint 可恢复：

- protocol snapshot 继续用旧键 `data.num_workers` 记录
  `train_num_workers`。
- `val_num_workers` 和 `test_num_workers` 只进入运行日志和普通 protocol
  metadata，不进入恢复不变量。
- 不修改已有 checkpoint，也不放宽其他协议字段的严格比较。
- 新旧 checkpoint 在 `train_num_workers=24` 时生成相同的
  `protocol_snapshot.data`。

## 日志与错误处理

- 训练启动日志同时打印 `train_num_workers`、`val_num_workers` 和
  `test_num_workers`。
- 运行记录保存三个实际值，便于复现内存和吞吐结果。
- 任一 worker 字段为负数、布尔值或非整数时，在构造 Dataset 和 Trainer
  之前失败，并在错误中指出字段名和值。
- DataLoader worker 内部异常继续传播到主进程，不增加静默单进程回退。

## 涉及文件

- `2.full_page_omr.ps1`
- `experiments/full_page_omr/config/ExperimentConfigWrapper.py`
- `experiments/full_page_omr/config/*/finetuning*.json`
- `experiments/full_page_omr/data.py`
- `experiments/full_page_omr/finetune.py`
- `tests/test_full_page_omr_config.py`
- `tests/test_full_page_omr_data_pipeline.py`
- `tests/test_full_page_omr_throughput.py`
- PowerShell 启动器相关测试

## 验收标准

1. 新配置可完成 JSON round-trip，输出只包含三个新 worker 字段。
2. 旧 `num_workers` 配置仍能加载为 `train_num_workers`，验证和测试默认为 0。
3. 新旧字段同时出现以及三个字段的非法值都会明确失败。
4. 三个 DataModule 的训练 DataLoader 使用 24 个持久 worker。
5. 验证和测试 DataLoader 使用 0 个 worker，且不启用持久 worker、
   `prefetch_factor` 或 worker 初始化函数。
6. PowerShell `-DryRun` 生成的临时配置明确包含 `24/0/0`。
7. 启动日志和运行 metadata 记录三个 worker 值。
8. 新配置构造的 protocol snapshot 仍为 `data.num_workers=24`，能够通过
   现有完整 checkpoint 的恢复协议比较。
9. 相关配置、DataLoader、throughput、resume 和 PowerShell 测试通过。
10. 本机短运行跨过一次验证边界；验证期间不新增 DataLoader 子进程，训练
    worker 在验证前后保持可复用。

## 非目标

- 本轮不重新寻找训练 worker 的最优值，训练默认保持 24。
- 不改变 batch size、预取深度、验证周期或 checkpoint 周期。
- 不通过继续扩大页面文件掩盖进程峰值。
- 不承诺验证耗时不变；同步验证的可控变慢是明确接受的取舍。
