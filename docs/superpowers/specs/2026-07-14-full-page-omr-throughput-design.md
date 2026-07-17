# Full-page OMR 训练吞吐与正确性修复规格

## 状态

实施版。已按论文、官方 MuSViT model card 和官方下游配置重新核对，并于 2026-07-14 锁定默认训练协议为 `fine_tune`。自动化修复与旧 checkpoint 反序列化 smoke test 已完成；24-worker、200-step 本机验收单独记录，不用环境结果替代协议正确性。

## 第一性原理

目标不是“让代码能跑”或“把显存占满”，而是在不伪造实验语义的前提下，缩短完成同样训练目标的墙钟时间，并保证配置、恢复、推理和错误处理可验证。

当前需要同时处理四类问题：

1. 实验协议含糊：代码名为 fine-tuning，却因为 `unfreeze_encoder()` 是空实现而始终执行 linear probing。
2. 空间尺度混淆：输入图像是 `1024 x 1024`，编码器输出却是 `64 x 64` patch 网格；下游 2D 位置编码错误地按输入像素分辨率分配。
3. 配置契约不一致：batch size、tokenization、worker 数和依赖锁文件存在多个相互矛盾的来源。
4. 已确认的正确性缺陷：配置序列化引用未定义函数、预测字典键类型不兼容、异常被静默吞掉。

现有吞吐剖析仍有效：

- 完整合成样本平均约 1.38 秒，主要耗时在 CairoSVG 和 Wand。
- 当前 checkpoint 约 1.51 GB，其中标准 `state_dict` 约 380 MB，优化器 tensor 约 45 MB。
- 当前训练每个 epoch 为 83 steps，约 2 分钟；逐 epoch 保存和逐 step `loss.item()` 都是可消除开销。
- 本机长时间实测已确认 `num_workers=24` 运行稳定，当前瓶颈不在 batch size。

## 外部事实与证据

### 论文协议

[MuSViT 论文 A.9.1](https://arxiv.org/html/2606.31811#Sx13.SSx9.SSSx1) 明确定义：

- Linear probing：encoder 全程冻结，只更新任务头。
- Fine-tuning：合成谱面阶段冻结 encoder；真实谱面开始加入后，解冻 encoder 并联合优化整个模型。

论文中 full-page OMR 的 frozen 结果和 fine-tuned 结果属于两张不同表，不能互换命名。这里的“只训练 decoder”按当前模型结构实际指任务头，即 `adaptor + autoregressive decoder`；encoder 之外的 adaptor 也必须训练。

### 官方输入尺度

[PRAIG/musvit model card](https://huggingface.co/PRAIG/musvit) 对整页谱面使用 `Resize([1024, 1024])`，输出 `4097` 个 token：一个 CLS token 加 `4096 = 64 x 64` 个 patch token，patch size 为 16。

因此：

- `1024 x 1024` 是 encoder 输入图像尺寸，与预训练分布对齐。
- `64 x 64` 是 encoder 输出的特征网格尺寸。
- 当前下游 `PositionalEncoding2D` 加在 adaptor 后的特征图上，所以它必须覆盖 `64 x 64`，不是 `1024 x 1024`。
- 把下游位置编码设为 `1024 x 1024` 不会保留原始乐谱物理尺寸；图像早已在预处理阶段缩放。

### Tokenization 来源

MuSViT model card 只定义视觉 encoder，不定义 full-page OMR 输出表示。tokenization 不能从视觉 model card 猜测。

[官方 Polish Scores 配置](https://github.com/OMR-PRAIG-UA-ES/MuSViT/blob/main/experiments/full_page_omr/config/Polish_Scores/finetuning.json) 使用 `tokenization_mode="bekern"`，并搭配 `Polish_Scores_BeKern` vocabulary；本仓库 Mozarteum 和 Polish Scores 的已跟踪配置也一致使用 BeKern。因此本实验默认输出表示确定为 `bekern`。

## 硬性需求

1. 训练协议必须显式为 `linear_probe` 或 `fine_tune`，日志、checkpoint 和运行脚本都能看出实际模式。
2. 整页输入默认保持 `1024 x 1024`；下游 2D 位置编码按实际 patch 网格生成，当前为 `64 x 64`。
3. `batch_size` 保持为 1，并在配置和 collate 边界拒绝其他值，不能继续静默丢图。
4. 生产配置的 `num_workers` 统一为 24，同时保留 `num_workers=0` 的同步回退路径。
5. tokenization 默认统一为 `bekern`；未知模式在加载数据或创建生成器前失败。
6. `pyproject.toml` 变化必须同步到 `uv.lock`，`uv lock --check` 必须通过。
7. 修复配置 round-trip、预测 token lookup 和静默异常。
8. 周期 checkpoint 默认每 100 epochs 保存一次；旧 checkpoint 仍可完整恢复。
9. 移除训练代码每 step 主动触发的 GPU 到 CPU 标量同步。

## 非目标

- 不实现 `batch_size > 1`；本轮只把 batch size 1 变成可执行不变量。
- 不把 Verovio、CairoSVG 或 Wand 移植到 GPU。
- 不实现多 GPU、DDP、梯度累积或模型并行。
- 不修改 BeKern 的符号语义、vocabulary 内容或 SER 计算方式。
- 不重写合成器视觉算法和课程概率公式。
- 不承诺 bit-for-bit 可复现；当前增强、随机选谱和原生渲染本就不是严格确定性的。

## 必须保持的语义

- `from_checkpoint` 是完整恢复，课程 step 为恢复后的 `trainer.global_step + skip_steps`。
- `starting_weights` 只加载权重，课程 step 从 `skip_steps` 开始。
- `num_workers=0` 不接收 `prefetch_factor` 或 `persistent_workers=True`。
- 无效 Verovio 节奏样本继续由生成器内部有限重试；超过上限后抛出明确异常。
- 验证集、测试集、学习率、精度模式和 best-on-`val_SER` checkpoint 保持不变。
- 旧 checkpoint 的 `i2w` 可能使用整数键；未来 JSON 来源可能使用字符串键，两者都要可预测。

## 设计

### 1. 显式训练协议

新增独立参数 `encoder_training_mode`，只接受：

```text
linear_probe
fine_tune
```

它不能复用现有 `finetuning` 参数，因为后者当前表示 `CL`、`SR`、`CL1` 等数据课程，不表示 encoder 是否训练。

规则：

- `linear_probe`：encoder 全程 `requires_grad=False`；adaptor 和 decoder 始终训练；删除任何按 global step 解冻的隐式分支。
- `fine_tune`：合成阶段冻结 encoder；真实数据首次可能进入训练分布时解冻，并保持解冻。
- `CL` 的真实数据从 `max_cl_steps = 120000` 后开始以非零概率进入，因此 fine-tune 解冻边界是 120000，不是当前硬编码的 200000。
- `SR` 的真实阶段从 `synth_pretraining_steps = 200000` 开始，因此其解冻边界是 200000。
- 解冻边界必须由数据课程暴露为单一来源，Trainer 不再复制魔法数字。
- `unfreeze_encoder()` 必须实际把 encoder 参数设为 `requires_grad=True`。
- optimizer 继续包含模型全部参数，以便冻结参数解冻后立即参与更新；不得在 optimizer 创建时永久过滤掉 encoder。
- 从 checkpoint 恢复到解冻边界之后时，必须在首个恢复 batch 的 forward 之前恢复正确的 `requires_grad` 状态；CL 的比较使用与数据课程一致的 `global_step + skip_steps`，不能让真实页已经进入而 encoder 仍冻结。
- checkpoint 记录 `encoder_training_mode`；恢复时显式传入的模式与 checkpoint 不一致则失败，不允许悄悄改变实验协议。
- 旧 checkpoint 没有该字段时，只允许调用者显式指定 `fine_tune` 完整恢复，并记录 legacy warning；`starting_weights` 仍可作为纯权重初始化跨协议使用。
- `CL1` 没有真实数据阶段，encoder 始终冻结；若它作为 fine-tune 流程的合成前置阶段运行，解冻只能发生在后续包含真实数据的 `CL` 或 `SR` 运行中。

`2.full_page_omr.ps1` 必须显式传该参数。本轮默认值已确定为 `fine_tune`，目标是论文 fine-tuning 结果；需要 frozen baseline 时必须显式改为 `linear_probe`。

### 2. 区分输入分辨率与特征网格

整页预处理默认继续使用 `1024 x 1024`。当前 MuSViT patch size 为 16，所以任务头特征网格为：

```text
feature_grid = input_resolution / patch_size = 1024 / 16 = 64
```

规则：

- `resolution` 必须为正整数，并能被 encoder patch size 整除。
- `PositionalEncoding2D` 构造函数只接收 embedding dim；第一次 forward 按实际特征图的 `H x W` 生成并缓存位置编码，shape、device 或 dtype 改变时重新生成。
- 缓存使用 `register_buffer(..., persistent=False)`；不得在构造函数里根据 `torch.cuda.is_available()` 私自选择设备。
- `maxh/maxw` 不再控制下游位置编码；本轮仅为旧 config 反序列化保留兼容字段，所有新代码和日志使用 `feature_grid_h/feature_grid_w`。
- forward 以移除 CLS 后的实际 token 数为最终真相，校验其为方形网格；当前标准输入必须得到 64。
- 当前 `1 x 256 x 1024 x 1024` 的 float32 表约占 1 GiB；正确的 `1 x 256 x 64 x 64` 约占 4 MiB。
- 位置编码不进入 checkpoint，避免纯确定性常量扩大文件。

### 3. Batch size 1 是契约，不是注释

当前 collate 使用 `images[0]`，而 decoder target 按 batch 维构造；在支持真正的图像 padding/stacking 前，batch size 大于 1 会静默训练错数据。

规则：

- 两份生产 JSON 配置和 PowerShell 入口保持 `batch_size=1`。
- 配置解析时若 `batch_size != 1`，抛出带字段名和值的 `ValueError`。
- `batch_preparation_img2seq` 若收到的样本数不是 1，再次 fail fast，防止直接构造 DataLoader 绕过配置校验。
- 本轮不实现大 batch；未来支持时必须重新设计可变尺寸图像 batching，而不是删除 guard。

### 4. Tokenization 统一为 BeKern

定义唯一支持集合：

```python
TOKENIZATION_MODES = {"kern", "ekern", "bekern"}
```

规则：

- `SyntheticOMRDataset`、`RealDataset`、`CurriculumTrainingDataset`、`SynthToRealDataset`、`VerovioGenerator` 和解析函数的默认值全部改为 `bekern`。
- 删除不存在语义的默认值 `standard`。
- 生产配置继续使用 `bekern`，并保持 vocabulary 名称中的 `BeKern` 对应关系。
- 入口在加载数据集、vocabulary 和 Verovio 前验证模式；未知值抛 `ValueError`，不能退化成只有 BOS/EOS 的序列。
- 三种模式的分支使用互斥 dispatch 或 `if/elif/else`，未知值必须进入错误分支。

### 5. Worker 默认值统一为 24

生产来源统一：

- `2.full_page_omr.ps1` 的 `Runtime.windows_num_workers = 24`。
- Mozarteum `finetuning.json` 的 `num_workers = 24`。
- Polish Scores `finetuning.json` 的 `num_workers = 24`。
- 验证 PowerShell 注入值的测试期望 24。

`num_workers=0` 和 `num_workers=2` 的针对性单元测试保留，因为它们分别验证同步回退和 Windows spawn；“统一 24”不等于篡改测试场景。

24 是本工作站已测得的生产默认值，不宣称是跨机器最优值。参数仍可覆盖，且必须验证为大于等于 0 的整数。

### 6. 更新依赖锁

`pyproject.toml` 已将 Hugging Face Hub 依赖改为带 `hf_xet` extra 的形式。实施时运行：

```powershell
uv lock
uv lock --check
```

规则：

- 提交 `uv.lock` 的对应变化，包括 `hf-xet` 依赖和 root package extra 元数据。
- 不手工编辑锁文件，不顺带升级无关依赖。
- 测试和训练命令使用锁定环境；锁文件不一致时直接失败。

### 7. 修复配置序列化 round-trip

`Data.to_dict()` 当前调用未定义的 `to_int()`。

规则：

- 不新增与 `from_int()` 完全重复的特殊分支；整数读写共用一个明确的类型校验函数，或统一命名为 `require_int()`。
- 不使用可被 `python -O` 删除的 `assert` 校验外部 JSON；错误类型使用 `TypeError` 或 `ValueError`，并指出字段。
- `skip_steps=0` 和非零值都必须满足 `from_dict(to_dict(data)) == data`。
- `experiment_config_to_dict()` 的结果必须可由 `json.dumps()` 序列化。

### 8. 修复 `predict(convert_to_str=True)`

问题的根源不是一次字典访问，而是同一份 `i2w` 允许两种键类型。模型初始化时把 `i2w` 一次性规范化为整数键；预测循环只处理一种数据结构，不在每个 token 上叠加 fallback 分支。

规则：

- `i2w` 的整数键原样保留，纯数字字符串键转换为整数；非数字键立即失败。
- 若 `1` 和 `"1"` 同时存在且映射不同 token，抛出冲突错误，不能任意覆盖。
- 预测出的 token id 始终保留为整数；EOS 判断和追加输出使用同一次整数键 lookup 的结果。
- `predicted_sequence` 中仍保存数值 id，返回的 `text_sequence` 仍为字符串列表，公开返回契约不变。
- `convert_to_str` 暂时保留为兼容参数并标记 deprecated，但不再改变内部 key 类型；传入 `True` 必须正常工作。
- 整数键和字符串键来源都覆盖测试；不存在的 token id 抛出包含 id 的明确 `KeyError`。

### 9. 异常必须可见，checkpoint 损坏不能伪装成成功

移除 `finetune.py` 中的 bare `except` 和 `except Exception: pass`。

训练结束后的 checkpoint 逻辑拆开：

- `best_model_path == ""` 是正常的“没有 best checkpoint”分支，不用异常控制流；保存 end checkpoint 后测试它。
- `best_model_path` 非空但加载失败，使用 `logger.exception(...)` 输出 checkpoint 路径、异常类型、消息和 traceback，然后重新抛出。仅打印后继续测试随机/旧权重会产生可信但错误的指标，因此禁止。
- 直接运行脚本时，若可选的 `musvit.env` 初始化失败，至少用 `logger.warning(...)` 输出异常类型和消息；该路径可继续使用已有环境变量，但不能静默。

### 10. Checkpoint 周期和体积

PowerShell 配置保留：

```powershell
$Config = @{
    checkpoint_every_n_epochs = 100
}
```

参数链必须完整贯通：

```text
2.full_page_omr.ps1
  -> --checkpoint_every_n_epochs
  -> experiments.full_page_omr.entrypoint.run
  -> experiments.full_page_omr.finetune.launch
  -> experiments.full_page_omr.finetune.main
  -> ModelCheckpoint(every_n_epochs=...)
```

规则：

- PowerShell 和 Python 入口都校验大于等于 1，默认均为 100。
- 文件名、目录、`save_top_k=1` 和 best validation checkpoint 策略不变。
- 从 epoch 1404 恢复时，下一次周期保存发生在 epoch 1500；按当前速度最坏恢复窗口约 3 小时 20 分钟，这是明确接受的策略。
- `SMTPP_Trainer.save_hyperparameters()` 忽略 `smt_model`；所有加载点显式传入 `smt_config` 和 `smt_model`。
- 旧 checkpoint 和新 checkpoint 都要可恢复；新 checkpoint 验收上限为 700 MB。

### 11. CPU 生产者与 GPU 消费者重叠

继续采用 PyTorch `DataLoader` 多进程和预取队列，不增加自定义线程池或第二套队列。

所有权规则：

- Dataset 不持有 Lightning `Trainer`。
- Dataset 跨进程只携带纯数据配置和共享 step counter。
- 每个 worker 惰性创建并独占 `VerovioGenerator`；pickle 状态不包含原生 toolkit。
- worker 异常由 DataLoader 抛回主进程，不允许静默退回单进程。
- `SharedStepCounter.reserve()` 为每个预取样本分配唯一课程 step；同一样本的 stage 和概率只使用该局部 step。
- `train_dataloader()` 在 worker iterator 创建前，以恢复后的 `global_step + skip_steps` 重置计数器。

`num_workers > 0` 时使用：

```text
persistent_workers=True
prefetch_factor=1
pin_memory=torch.cuda.is_available()
```

worker 初始化函数在模块顶层，使用 `torch.initial_seed()` 同时初始化 Python `random` 和 NumPy。

多 worker 预取会使课程边界存在至多 `num_workers * prefetch_factor` 个在途样本。24 workers 下最坏为 24 个样本，相对 40,000-step 阶段长度可接受；测试和文档必须同步采用该上限，不能沿用旧值。

### 12. 消除每 step 的应用级同步

删除无消费者的 best/worst loss 与 image 状态，以及 `training_step` 中全部 `loss.item()` 分支。

训练 loss 只做 epoch 聚合：

```python
self.log(
    "loss",
    loss,
    on_step=False,
    on_epoch=True,
    prog_bar=True,
    batch_size=x.shape[0],
)
```

这只保证应用代码不再每 step 主动读取 GPU 标量；不宣称 Lightning、AMP 或 optimizer 内部零同步。

## 涉及文件

- `2.full_page_omr.ps1`
- `pyproject.toml`
- `uv.lock`
- `experiments/full_page_omr/config/ExperimentConfigWrapper.py`
- `experiments/full_page_omr/config/Mozarteum/finetuning.json`
- `experiments/full_page_omr/config/Polish_Scores/finetuning.json`
- `experiments/full_page_omr/entrypoint.py`
- `experiments/full_page_omr/finetune.py`
- `experiments/full_page_omr/data.py`
- `experiments/full_page_omr/Generator/SynthGenerator.py`
- `experiments/full_page_omr/smt_foundation/modeling_smt.py`
- `experiments/full_page_omr/smt_foundation/configuration_smt.py`
- `experiments/full_page_omr/smt_trainer.py`
- `tests/` 下对应的配置、模型、checkpoint、spawn DataLoader 和训练协议测试

## 验收标准

### 自动化测试

1. `linear_probe` 下任意 step 的 encoder 参数均冻结，adaptor/decoder 可训练；`fine_tune` 在课程边界前冻结、边界后解冻。
2. 从边界后的 checkpoint 恢复时，第一个 forward 前 encoder 已处于正确状态；模式不匹配恢复失败。
3. 1024 输入产生 4096 个非 CLS token 和 `64 x 64` 特征网格；2D PE 容量为该网格，模型构造不再额外占约 1 GiB。
4. `batch_size=1` 正常；配置为 2 或 collate 收到两个样本时明确失败，不再丢弃第二张图。
5. 所有 dataset/generator 默认 tokenization 为 `bekern`；`standard` 和任意未知值在重资源初始化前失败。
6. 两份生产 JSON 和 PowerShell 默认 worker 均为 24；0-worker 回退和 2-worker spawn 测试仍保留。
7. `Data` 与 `ExperimentConfig` 对含非零 `skip_steps` 的配置完成 JSON round-trip。
8. `predict` 对整数键和字符串键 vocabulary 均能识别 EOS，且 `convert_to_str=True` 不再 KeyError。
9. 无 best checkpoint 时保存并测试 end checkpoint；best checkpoint 损坏时记录异常并让测试失败。
10. `uv lock --check` 通过，锁文件包含 `hf-xet` extra，不升级无关包。
11. PowerShell 到 `ModelCheckpoint.every_n_epochs` 的参数链测试通过，默认 100，非法值启动前失败。
12. `training_step` 不调用 `Tensor.item()`，不维护 best/worst image 状态，只记录 epoch loss。
13. Windows spawn 以多 worker 跨两个短 epoch 取样，不出现 toolkit 或 Trainer pickle 错误。
14. 恢复 checkpoint 后首个预取样本使用恢复 step；`skip_steps` 只加一次。
15. 现有 epoch 1404 checkpoint 和新 checkpoint 都能恢复，新 checkpoint 不序列化 `smt_model`。
16. 现有无效节奏样本重试测试继续通过。

### 本机运行验收

- 使用 `batch_size=1`、`num_workers=24` 连续运行至少 200 steps，排除 worker 首次初始化后记录 samples/s、step 中位数、GPU/CPU 利用率和峰值内存。
- loss 必须有限，global step 每 batch 加一，课程阶段在最多 24 个在途样本的边界误差内不回退。
- 日志必须打印 `encoder_training_mode`、输入 resolution、feature grid、tokenization mode、batch size 和 worker 数，保证实验结果可追溯。
- 吞吐验收以相同训练协议下的 samples/s 为准，不以显存占用率为目标。

## 实施顺序

本节记录已批准并执行的顺序：

1. 先确认 `2.full_page_omr.ps1` 的 `encoder_training_mode` 默认值。
2. 修复训练协议、2D PE、batch guard、tokenization、配置 round-trip、predict 和异常处理，并运行正确性测试。
3. 统一 worker=24 和依赖锁，运行配置与锁文件测试。
4. 完成 checkpoint、DataLoader 和同步优化的剩余实现，运行 Windows spawn、恢复和旧 checkpoint 测试。
5. 最后做 200-step 本机运行验收；性能数字不能替代协议正确性。

## 实施验证记录（2026-07-14）

- `uv lock --check` 通过；`hf-xet` extra 已进入锁文件，未升级无关依赖。
- 自动化测试覆盖训练协议、checkpoint 元数据、惰性 2D PE、配置 round-trip、batch guard、tokenization、预测键、Verovio 原生错误和 Windows spawn；2026-07-17 提交前隔离复验为 74 passed、19 subtests passed。
- PowerShell dry run 明确输出 `encoder_training_mode=fine_tune`、`num_workers=24`、checkpoint interval 100。
- 当前可用 legacy checkpoint 在本次验收启动时为 step 107900；完整恢复成功，encoder 保持冻结。独立边界 smoke test 验证 step 116532 仍冻结、step 120000 解冻。
- 24 workers 从 step 107900 连续训练到 108100，共 200 steps；排除冷启动后的耗时为 72.73 秒，吞吐 2.75 samples/s，抽样 loss 全部有限（0.288 至 1.061）。
- PyTorch 峰值显存 allocated/reserved 为 826.8/938.0 MiB；物理 GPU 0 峰值利用率 60%。24 个 worker 峰值 working set 28.11 GiB、private bytes 71.21 GiB，系统最低空闲内存 33.72 GiB。
- 新完整 checkpoint 为 424,879,099 bytes（约 405 MiB），包含 `fine_tune/120000`，不序列化 `smt_model`，满足小于 700 MB 的约束。
- 已删除真实数据集的 eager `self.x/self.y` 图像与标签列表。`RealDataset`、`CurriculumTrainingDataset` 和 `SynthToRealDataset` 现在只持有 Hugging Face `MemoryMappedTable`，worker 在 `__getitem__` 中按 Arrow 行解码、tokenize，并继续在线执行原有 resize；本轮没有增加缩放图缓存。
- Polish Scores 训练 split 实测为 83 行，Arrow source pickle 为 2,121 bytes，包含 vocabulary 的完整 `RealDataset` pickle 为 5,847 bytes。24 个 worker 在约 7 秒内全部创建，纯真实数据首 batch 为 60.39 秒；原 eager 列表路径约 15 分钟的逐 worker 复制已消除。
- 当前 CL checkpoint 在验收时为 step 124500。GPU0、24-worker 实际恢复训练的首 batch 为 102.62 秒，随后 20 steps 用时 27.53 秒，抽样 loss 为 0.364209 和 0.374406，均有限；encoder 因已越过 120000 边界而在首个 forward 前解冻。
- 该 CL smoke test 的物理 GPU0 峰值利用率为 94%，峰值显存 10,877 MiB；24 个 worker 峰值 working set/private bytes 为 24.83/67.53 GiB，系统最低空闲内存 25.08 GiB。剩余冷启动主要来自 Windows worker 导入训练栈，以及每个 worker 惰性初始化在线 Verovio 数据源，不再来自完整真实图像列表复制。
- 计划指定的 `polish_scores_cl_CL-epoch1404.ckpt` 在最终验收时已不在工作区，因此不能声称完成该文件的身份级恢复；改用当前真实 legacy checkpoint 验证了相同恢复路径。
