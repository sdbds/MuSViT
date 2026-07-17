# Full-page OMR 规格 Linus 复审

## 三个问题

1. 这是实际问题吗？是。空解冻函数改变实验含义，错误尺寸的位置编码浪费约 1 GiB，未知 tokenization 会生成空监督，另外三个缺陷都有可触发的运行时失败。
2. 有更简单的办法吗？有。训练协议显式二选一、`i2w` 在模型边界一次规范化、位置编码按实际 feature map 惰性缓存；不要在训练循环和预测循环里继续堆特殊分支。
3. 会破坏什么？主要风险是旧 checkpoint 没有训练模式、现有调用仍传 `convert_to_str`、以及 worker=24 扩大课程边界的在途样本数。修订规格分别提供显式 legacy mode、兼容参数和 24-sample 边界契约。

## 结论

通过。规格已覆盖用户提出的九项决定，并消除了上一版中 worker=2、2D PE 使用输入像素尺度、`standard` tokenization 和静默异常等错误假设。默认协议随后锁定为 `fine_tune`，PowerShell、Python 入口、日志和 checkpoint 使用同一显式值，不再存在实验级决策阻断。

## Findings

### [P1] “只训练 decoder”不是普遍正确，只对 linear probing 正确

论文 A.9.1 同时定义了两套 full-page OMR 协议：frozen encoder 的 linear probing，以及真实谱面加入后解冻 encoder 的 fine-tuning。当前 `finetune.py`、实验名和空的 `unfreeze_encoder()` 互相矛盾。修订规格正确地增加独立 `encoder_training_mode`，并把 CL/SR 的解冻边界交还给数据课程；但脚本默认值仍必须由目标表格决定。

若目标是验证 frozen representation，选 `linear_probe`，对应论文 Table 15。若目标是复现论文最佳 full-page OMR 结果，选 `fine_tune`，对应 Table 16；继续冻结会系统性偏离目标，而不是“更稳的 fine-tuning”。

### [P1] 1024 和 64 属于两种不同的数据结构

1024 是预处理后的图像边长，64 是 patch feature map 边长。把像素尺寸传给下游位置编码会分配约 1 GiB、实际只使用前 64 行和 64 列。修订规格改为按 forward 的实际特征图惰性生成非持久 buffer，直接消除了错误状态，而不是把常量从 1024 随手改成另一个魔法数。

### [P1] 仅打印 checkpoint 异常后继续，仍然是错误实现

best checkpoint 非空却加载失败，通常意味着文件损坏、模型契约不兼容或路径错误。打印后改用当前内存权重继续测试会生成看似有效但来源错误的指标。规格正确区分“没有 best checkpoint”这一正常分支与“存在但加载失败”这一异常分支，后者记录 traceback 后重新抛出。

### [P1] batch size 1 必须 fail fast

当前 collate 的 `images[0]` 会在 batch 大于 1 时静默丢图。用户决定暂不提高 batch size 是合理的，但仅在 JSON 里写 1 不够。规格同时要求配置边界和 collate 边界拒绝其他值，才能把已知限制变成可靠契约。

### [P2] model card 不能回答输出 tokenization

官方 model card 只描述 MuSViT 视觉 encoder。BeKern 的依据来自官方 full-page OMR 配置、已跟踪 vocabulary 和生成器实现。规格将默认值统一为 `bekern` 并拒绝未知模式，避免原来的 `standard` 路径退化成只有 BOS/EOS 的空监督。

### [P2] `i2w` 应在边界规范化，不应在预测循环里补特殊分支

官方 `.npy` vocabulary 使用整数键，而 JSON 可能把键转成字符串。修订规格要求模型初始化时统一为整数键并检测冲突，`predict()` 只处理一种结构。保留 `convert_to_str` 仅用于调用兼容，优于每个 token 尝试两次字典访问。

### [P2] worker=24 改变了课程边界误差上限

上一版按两个 worker 声称最多两个在途样本，改成 24 后该结论失效。修订规格已把上限同步为 `24 * prefetch_factor = 24`，同时保留 0-worker 回退和 2-worker spawn 测试。24 是本机生产默认值，不是库级真理，继续保留配置覆盖是正确的。

### [P2] 旧 checkpoint 缺少训练模式，不能靠文件名猜

旧 checkpoint 没有 `encoder_training_mode`。规格要求恢复时由调用者显式指定，并记录 legacy assignment；新 checkpoint 则校验模式一致性。这既保住 epoch 1404 checkpoint 的兼容性，也避免恢复后悄悄改变 encoder 训练状态。

## 其余审查结果

- `uv.lock` 更新被写成可验证交付物，要求 `uv lock --check`，没有把环境漂移留给运行时。
- `Data.to_dict()` 不再通过新增同义 helper 掩盖问题；规格要求统一外部配置验证并覆盖 JSON round-trip。
- checkpoint 周期、体积、旧恢复、Windows spawn、课程 step 和逐 step GPU 同步都有明确验收项。
- “统一 worker=24”只作用于生产默认和对应期望，不篡改用于验证 0/2 worker 行为的测试场景。

## 决策结果

实施前只需要确认一件事：

```text
linear_probe -> encoder 全程冻结，目标是论文 frozen baseline
fine_tune    -> 合成阶段冻结，真实页加入后解冻，目标是论文最佳结果
```

本轮选择 `fine_tune`：CL 在 step 120000、SR 在 step 200000 解冻，CL1 始终冻结。`linear_probe` 保留为显式可选 baseline，不共享默认值。

## 实施后复审

正确性修复通过自动化测试和真实 checkpoint 恢复。200-step 验收确认新 checkpoint 约 405 MiB、不含 `smt_model`，恢复后的 encoder 在课程边界前保持冻结，loss 有限。

Windows spawn 复制 eager 真实图像列表的问题已经按更简单的数据结构修复：三个真实数据集只保留 Hugging Face Arrow `MemoryMappedTable`，每个 worker 在取样时按行解码、tokenize 和执行原有 resize。真实数据集的 pickle 已降至约 5.8 KiB，24 个 worker 约 7 秒内全部创建，纯真实数据首 batch 从约 15 分钟降至 60.39 秒；没有引入共享内存协议，也没有提前缓存缩放图。

实际 CL 恢复的首 batch 仍需 102.62 秒。这不是 Arrow 改造失败，而是瓶颈已经移动到 24 个 Windows 进程导入 Python/训练依赖，以及每个 worker 首次需要合成样本时各自惰性加载 Verovio 数据源。下一步若继续优化，应先剖析并减少合成数据源的每 worker 初始化成本；此时再做缩放图缓存只能减少稳态 CPU resize，不能解决这段冷启动。

另一个环境风险是 CUDA ordinal 与 `nvidia-smi` 编号在这台双卡机器上相反；本次 `CUDA_VISIBLE_DEVICES=1` 实际运行于物理 GPU 0（RTX 5090）。若脚本需要按 `nvidia-smi` 编号选卡，应先固定 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 或使用 GPU UUID，不能把裸数字当成稳定硬件身份。

## 2026-07-17 提交前复审

提交前发现并修复一个协议级问题：CL 数据课程使用 `global_step + skip_steps`，encoder 原先却只按裸 `global_step` 解冻。非零 `skip_steps` 会让真实页已经进入训练而 encoder 继续冻结。现在 `curriculum_step_offset` 由数据课程显式提供、参与解冻比较并写入 checkpoint；恢复时 offset 不一致会失败。

同时排除了 PowerShell 中本机 `epoch3500.ckpt`、CUDA ordinal 和 Cairo 路径覆盖，避免把不可移植的运行现场当成仓库默认值提交。最终隔离环境测试为 74 passed、19 subtests passed；未发现剩余提交阻断项。残余风险仍是 24 个 Windows worker 的 Verovio 首次初始化较重，以及 GPU ordinal 的机器相关性，两者均已显式记录，不影响本次正确性提交。
