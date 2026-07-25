# Staff-level OMR 可信训练协议优化设计

日期：2026-07-25

状态：待评审

目标协议：`staff_omr_v2`

## 1. 背景

`experiments/staff_level_omr` 已具备可运行的 MuSViT + BiLSTM + CTC
训练原型，支持 linear probing 和 LoRA。当前实现缺少稳定的数据、输入、
checkpoint 和恢复契约，部分错误会在训练过程中被静默掩盖。

本设计先解决实验可信度。吞吐优化、优化器调整和模型结构消融留到可信基线
建立之后，避免同时改变多个变量而无法解释 CER 变化。

## 2. 决策

本期建立一个版本化的 staff-level OMR 训练协议，满足以下条件：

1. 每个训练样本要么产生合法的 CTC 梯度，要么在加载模型前触发明确错误。
2. 数据划分、词表、输入几何和基础模型具有稳定身份。
3. checkpoint 能独立恢复训练，并包含把预测 id 解码为符号所需的信息。
4. 同一配置的重复运行不会覆盖彼此的产物。
5. 评估结果能够说明多少样本受输出长度上限影响。
6. 关键契约由不依赖私有数据、网络和 GPU 的测试覆盖。

## 3. 本期范围

本期包含：

- 单一、经过校验的训练配置。
- 固定且可审计的数据 split manifest。
- 版本化词表。
- CTC 可行性预检和分层评估。
- 变长 target 数据管线。
- 统一的输入几何契约。
- 完整、原子、可恢复的 checkpoint。
- 明确的训练生命周期和失败行为。
- CLI 兼容层和 README 校正。
- CPU、本地、无网络的契约测试。

本期不包含：

- AMP 或其他精度模式调整。
- AdamW、WSD、cosine schedule 或其他优化器协议变更。
- 梯度裁剪策略变更。
- DataLoader 吞吐调优。
- `mean`、flatten、attention pooling 等行轴聚合消融。
- beam search 或语言模型解码。
- ONNX 或其他部署格式。
- PowerShell launcher 的实现或改造。
- 自动推断 `group_id` 或生成 split manifest 的迁移工具。
- 把 metadata 不完整的 legacy checkpoint 转换为 v2。

`num_workers`、`max_epochs`、`start_eval` 和 `patience` 会进入统一配置，
目的是消除硬编码并支持校验，不在本期宣称性能收益。

## 4. 审查裁决

### 4.1 纳入本期

| 问题 | 裁决 | 本期处理 |
|---|---|---|
| CTC 只检查 `target_length <= T` | 正确性缺陷 | 使用 `L + adjacent_repeats <= T` 预检 |
| `zero_infinity=True` 掩盖无效样本 | 静默失败 | 训练预检通过后禁用掩盖 |
| checkpoint 不保存词表 | 产物不可独立解释 | 保存完整词表和模型契约 |
| `start_eval > max_epochs` 最后才失败 | 参数校验缺失 | 启动前拒绝无评估配置 |
| checkpoint 文件名覆盖 | 数据丢失风险 | 每次运行使用独立目录和 run id |
| 评估通过 blank 哨兵截断 target | 脆弱的数据契约 | 使用 batch 中的真实 target lengths |
| `64` 和 `1024` 硬编码 | 输入契约错误 | v2 使用 exact-grid；基础尺寸从 backbone config 读取 |
| 两套 `shape_patches` 语法 | CLI 双重事实来源 | 统一公共参数并保留过渡别名 |
| README 与代码不一致 | 用户可见行为错误 | 文档由新契约校正 |
| epoch、patience、workers 硬编码 | 配置不可审计 | 纳入单一配置并验证 |

### 4.2 接受问题，调整处理方式

训练样本不再由 `filter_max_len` 自动丢弃。自动过滤会改变训练分布，单纯打印
数量仍不足以建立可信协议。v2 对训练 split 中任何 CTC 不可行样本执行
fail-fast，并列出样本 id、所需时间步、可用时间步和汇总比例。需要排除样本时，
排除决定必须写进版本化 manifest，不能由训练循环临时决定。

验证和测试 split 中的不可行样本保留。它们反映模型时间轴容量的真实限制，
但必须同时报告全量 CER、可行子集 CER、不可行样本数量和比例。跨 patch
宽度比较时，以全量 CER 为主指标，并同时展示容量统计。

词表不应在每次训练时从 test split 临时 fit。对于 closed-vocabulary OMR，
词表可以来自完整语料，但必须先生成独立、版本化的词表文件，并在协议中声明
其来源。运行时不得通过读取当前 test 内容来改变输出 head。

### 4.3 延后并要求证据

以下建议合理，但不能写成无条件收益：

- AMP 可能提高 ViT 吞吐并降低显存，实际收益和数值等价性依赖 GPU、PyTorch
  版本和 batch shape。后续需要吞吐、峰值显存和 CER parity 基准。
- `pin_memory`、`persistent_workers`、`prefetch_factor` 和 worker 数量需要按
  Windows/Linux 分开测量。`full_page_omr` 只为部分训练 DataLoader 启用
  persistent workers，val/test 并非全部启用。
- warmup、cosine 或 WSD 不是免费的 CER 改善。它们会改变优化协议，需要固定
  split、样本数和随机种子后单独比较。
- 梯度裁剪对 LSTM + CTC 可能有价值，但当前 `full_page_omr` Trainer 配置没有
  显式启用梯度裁剪，不能把它列为已经验证的仓库惯例。
- 行轴均值池化可能限制音高信息，也可能依靠 ViT 的位置编码保留足够线索。
  这是模型假设，不是已经确认的 bug，必须通过受控消融判断。

### 4.4 不采纳的表述

- `LabelEncoder.classes_` 的顺序不依赖 `os.listdir` 顺序；scikit-learn 会对
  类别排序。真正的问题是词表没有保存，以及 split 身份受无序文件枚举影响。
- CTC `input_lengths` 不能简单地在循环外创建一次，因为最后一个 batch
  可能具有不同大小。可以按当前 batch 构造或按 batch size 缓存，但这不是
  本期正确性重点。
- “AMP 固定获得 1.5-2x”及“warmup + cosine 基本白送 CER”缺少本项目测量，
  不作为需求或验收标准。

## 5. 版本和兼容边界

### 5.1 协议版本

新训练运行写入：

```text
protocol_version = staff_omr_v2
```

当前实现产生的权重视为：

```text
protocol_version = staff_omr_legacy_v1
```

v2 loader 一律拒绝 legacy 权重。legacy 文件缺少词表、split、预处理和优化器
状态，无法无损转换。需要复查旧结果时，应使用产生该 checkpoint 的 legacy
代码版本和外部记录；该行为不属于 v2。

### 5.2 CLI 兼容

训练方法的规范名称为：

```text
linear_probe
lora
```

现有 `linear_prob` 在一个兼容周期内作为别名接受，运行记录只保存规范名称。

patch grid 的规范参数为两个独立整数：

```text
--patch_rows=8
--patch_cols=128
```

现有 `--shape_patches` 作为过渡别名。若新旧参数同时出现，启动时拒绝运行，
不猜测优先级。直接模块入口和 `musvit staff-level-omr` 必须调用同一个配置
解析与校验函数。

### 5.3 结果兼容

v2 改变 split 和 linear probe 输入契约，因此旧 CER 不得与 v2 CER 放入同一
结果序列。报告必须展示协议版本、split hash、vocabulary hash 和 input
contract。旧结果可作为历史参考，但不能作为同协议回归基线。

## 6. 单一训练配置

引入一个不可变、经过验证的 `StaffOMRConfig`。CLI、直接 Python 调用和
checkpoint 恢复都使用这一结构。

配置至少包含：

| 字段 | 约束 |
|---|---|
| `experiment_name` | 非空；只用于人类识别，不作为唯一目录名 |
| `dataset_id` | 非空稳定标识 |
| `data_path` | 存在且可读 |
| `split_manifest_path` | 存在且符合 manifest schema |
| `vocabulary_path` | 存在且符合 vocabulary schema |
| `model_name` | 受支持的 MuSViT alias |
| `model_revision` | Hugging Face 不可变 commit SHA |
| `method` | `linear_probe` 或 `lora` |
| `patch_rows` | 正整数 |
| `patch_cols` | 正整数 |
| `batch_size` | 正整数 |
| `num_workers` | 非负整数 |
| `learning_rate` | 有限正数 |
| `max_epochs` | 正整数 |
| `start_eval` | `1 <= start_eval <= max_epochs` |
| `patience` | 正整数 |
| `seed` | 非负整数 |
| `output_root` | 可创建目录 |
| `device` | 明确的 `cpu`、`cuda` 或经过记录的 `auto` |

`experiment_name`、`dataset_id`、`data_path`、`split_manifest_path`、
`vocabulary_path` 和 `model_revision` 没有默认值，v2 调用者必须显式提供。
其余默认值保持当前公开训练行为：

```text
model_name = musvit
method = lora
patch_rows = 8
patch_cols = 64
batch_size = 8
num_workers = 6
learning_rate = 3e-4
max_epochs = 1000
start_eval = 20
patience = 30
seed = 7
output_root = experiments/staff_level_omr/runs
device = cuda
```

配置校验必须在下载或加载 backbone、创建 CUDA context、启动 DataLoader
workers 之前完成。

配置还固定记录当前优化协议：

```text
optimizer = torch.optim.Adam
betas = [0.9, 0.999]
eps = 1e-8
weight_decay = 0.0
scheduler = none
```

learning rate 使用配置值并保持恒定。v2 不引入 scheduler。

配置解析后生成 canonical JSON。键排序、数值类型和规范名称固定，其 SHA-256
写入 run metadata 和 checkpoint。resume 要求 canonical config SHA-256 完全
一致，不设置含义不明的 immutable fields 子集。

所有 canonical JSON hash 使用同一序列化规则：UTF-8、对象键按字典序、
`ensure_ascii=false`、分隔符 `,` 和 `:` 周围无空格、无结尾换行。manifest 的
samples 在序列化前按 `sample_id` 排序，路径统一为 `/` 分隔的相对路径；
vocabulary 的 tokens 数组顺序具有语义，不排序。

## 7. 数据和 split manifest

### 7.1 Manifest schema

v2 不在训练启动时调用 `train_test_split`。训练需要一个持久化 manifest：

```json
{
  "schema_version": "staff_omr_split_v1",
  "dataset_id": "catedrales",
  "group_semantics": "score_or_page",
  "samples": [
    {
      "sample_id": "score01-page03-staff02",
      "group_id": "score01-page03",
      "image_path": "score01-page03-staff02_region.png",
      "image_sha256": "<64 lowercase hex chars>",
      "target_path": "score01-page03-staff02_gt.txt",
      "target_sha256": "<64 lowercase hex chars>",
      "split": "train"
    }
  ]
}
```

路径相对 `data_path` 解析。`sample_id` 和 `group_id` 必须非空且稳定。

### 7.2 Manifest 校验

启动前验证：

1. schema version 受支持。
2. `sample_id` 唯一。
3. 每个 `group_id` 只出现在一个 split。
4. split 只允许 `train`、`val`、`test`。
5. 三个 split 均非空。
6. image 和 target 文件存在且可读。
7. image 和 target 的内容 SHA-256 与 manifest 一致。
8. target 解析后非空。
9. 同一文件不能被多个 sample 引用。
10. 所有路径解析后仍位于 `data_path` 内。

manifest 以 `sample_id` 排序后规范化并计算 SHA-256。训练不修改 manifest。

### 7.3 旧目录迁移

只有平铺的 `*_region.png` 和 `*_gt.txt` 时，数据维护者必须在训练前提供
manifest。训练命令不自动随机切分，也不猜测文件名中的作品或页面身份。
manifest 生成工具不在本期范围；手工或外部工具生成的文件都必须通过同一
schema、group 隔离和内容 hash 校验。

## 8. 词表契约

词表使用独立 JSON：

```json
{
  "schema_version": "staff_omr_vocab_v1",
  "dataset_id": "catedrales",
  "vocabulary_scope": "closed_corpus",
  "source_manifest_sha256": "<64 lowercase hex chars>",
  "blank_id": 0,
  "tokens": ["token-a", "token-b"]
}
```

`tokens[0]` 映射到 id 1，blank 固定为 id 0。token 不得为空或重复。

运行前验证所有 split 的 token 都存在于词表。出现 OOV 时列出 split、样本 id
和 token 计数，然后终止。运行时不重新 fit `LabelEncoder`。

v2 首版只接受 `vocabulary_scope = closed_corpus`。词表生成工具不在本期范围。
`source_manifest_sha256` 必须等于本次运行的 manifest hash。词表的 canonical
SHA-256 写入 run metadata 和 checkpoint。

## 9. CTC 可行性契约

目标 id 序列 `y` 的最小 CTC 时间步为：

```text
minimum_frames(y) =
    len(y)
    + count(y[i] == y[i - 1] for i in 1..len(y)-1)
```

模型可用时间步为 `patch_cols`。预检对 train、val、test 全部执行。

### 9.1 Train

若任何训练样本满足：

```text
minimum_frames(target) > patch_cols
```

则在加载 backbone 前终止。错误报告包含：

- 不可行样本数和占 train 的比例。
- 最大 required frames。
- 每个失败样本的 `sample_id`、target length、adjacent repeats、required frames
  和 available frames。

v2 不在训练循环内丢弃这些样本。CTC loss 使用 `zero_infinity=False`，任何
非有限 loss 都作为协议违反立即报错。

### 9.2 Validation 和 test

不可行样本保留，并在 sample metadata 中标记。评估同时输出：

```text
val_CER_all
val_CER_feasible
val_infeasible_samples
val_infeasible_ratio

test_CER_all
test_CER_feasible
test_infeasible_samples
test_infeasible_ratio
```

`val_CER_all` 是 checkpoint 选择指标。`val_CER_feasible` 用于解释容量影响，
不能替代全量指标。validation 和 test 只执行解码与 CER 计算，不计算 CTC
loss。若某个 split 没有可行样本，其 `CER_feasible` 写为 JSON `null`，同时
记录 feasible sample count 为 0。

## 10. Target 数据管线

Dataset 返回：

```text
image_tensor
target_ids_1d
target_length
sample_id
ctc_feasible
```

Dataset 不执行全局 padding。训练 collate 将一个 batch 的 target 拼接为 CTC
支持的一维 tensor，并返回每个 target 的真实长度。评估使用真实长度切片，
不再通过查找第一个 blank 恢复 target。

`input_lengths` 按当前输出 tensor 的 batch size 和时间维生成。实现可以使用
CPU long tensor或按 batch size 缓存，但必须正确处理最后一个较小 batch。

## 11. 输入和模型契约

### 11.1 Exact-grid

v2 的 linear probe 和 LoRA 使用同一图像尺寸：

```text
height = patch_rows * backbone_patch_height
width  = patch_cols * backbone_patch_width
```

两种方法都调用位置编码插值。`method` 只决定 backbone 的可训练参数，不决定
resize、padding 或 reshape 行为。

预处理固定为：

1. 解码为三通道 RGB。
2. 直接 resize 到 exact-grid 尺寸，不保持原宽高比，不执行 crop 或 pad。
3. 使用 bilinear interpolation，并明确启用 antialias。
4. 使用 `ToTensor` 语义缩放到 `[0, 1]`，不做 mean/std normalization。

这些字段写入 `input_contract`，不能依赖 TorchVision 的隐式默认值。

### 11.2 Backbone metadata

patch size、hidden size、prefix token 数和原始 image size 从加载后的 backbone
config 读取并校验。alias 配置只保存模型 id 和固定 revision，不复制可能漂移
的维度常量。

删除 v2 路径中的 `64x64` reshape 和 `1024` padding 假设。backbone 输出移除
prefix token 后必须严格满足：

```text
spatial_token_count == patch_rows * patch_cols
```

不满足时抛出包含模型 id、revision、输入尺寸、patch size 和实际 token 数的
错误。

### 11.3 本期保持的模型结构

本期保持 projection、两层 bidirectional LSTM、classifier 和行轴 mean pooling
不变。这样 v2 首先回答训练协议是否可信，不把模型结构变化混进基线。

LoRA 参数保持当前协议并写入 canonical config：

```text
rank = 8
alpha = 16
dropout = 0.1
bias = none
target_modules = [query, key, value]
use_rslora = true
```

`task_head` 明确定义为 projection、RNN 和 CTC classifier。linear probe 训练
整个 task head；LoRA 训练 adapter 和整个 task head。

## 12. 训练生命周期

训练按以下顺序执行：

1. 解析并校验配置。
2. 加载并校验 manifest 和词表。
3. 读取 target，完成 OOV、split 和 CTC 预检。
4. 写入 run metadata 初始记录。
5. 加载固定 revision 的 backbone。
6. 校验输入和 token geometry。
7. 构建 Dataset、DataLoader、模型和优化器。
8. 训练并按配置执行验证。
9. 每个完整 epoch 结束后原子更新 `last`；验证严格改善时原子更新 `best`。
10. 从 `best` checkpoint 执行 test。
11. 写入最终 run summary。

### 12.1 Epoch 和 early stopping

`max_epochs`、`start_eval` 和 `patience` 不再硬编码。

必须满足：

```text
1 <= start_eval <= max_epochs
patience >= 1
```

从 `start_eval` 起每个 epoch 执行一次 validation。`patience` 统计连续多少次
已执行的 validation 没有严格改善 `val_CER_all`；达到阈值后停止。

验证指标为 NaN 或 infinity 时立即失败，不能把既有旧 checkpoint 当作本次
训练结果。每次运行使用空的新 run 目录。

### 12.2 Resume 边界

v2 只承诺 epoch 边界的 state-complete resume。checkpoint 保存模型、优化器、
early-stopping、Python、NumPy、Torch CPU、Torch CUDA 和 DataLoader generator
状态。恢复后从下一个 epoch 开始，不承诺 batch 中间恢复。

resume 前校验以下字段完全一致：

- protocol version
- canonical config SHA-256
- vocabulary hash
- split manifest hash
- 实际 device type
- package versions
- PyTorch deterministic algorithm 和 cuDNN flags

不一致时拒绝 resume。v2 首版不支持在 resume 时更换任何 canonical config
字段。这里的 state-complete 指所有继续训练所需状态都已恢复；跨硬件、驱动或
依赖版本不承诺 bitwise 等价。CPU dummy integration test 在相同环境中要求
恢复后的下一 epoch 与不中断运行 tensor 完全一致。

## 13. 运行目录和 checkpoint

输出目录结构：

```text
<output_root>/
  <experiment_name>/
    <timestamp>-<config_hash_prefix>-<run_id>/
      run.json
      preflight.json
      metrics.jsonl
      checkpoints/
        last.pt
        best.pt
      summary.json
```

`run_id` 使用完整 `uuid4().hex`。run 目录通过 `mkdir(exist_ok=False)` 创建；
已存在即失败，不覆盖也不重试猜测。所有文件先落到同目录临时文件，再用原子
replace 发布。

checkpoint 至少包含：

```text
schema_version
protocol_version
run_id
model_state_dict
optimizer_state_dict
epoch
global_step
early_stopping_state
best_metric_name
best_metric_value
config
config_sha256
split_manifest
split_manifest_sha256
vocabulary
vocabulary_sha256
base_model_id
base_model_revision
backbone_config
input_contract
rng_state
package_versions
```

`best.pt` 和 `last.pt` 都使用相同 schema。checkpoint loader 必须先验证 metadata，
再把 tensor state 应用到已构建模型。`summary.json` 保存 `best.pt` 和 `last.pt`
的文件 SHA-256，使 checkpoint 离开原路径后仍可核对内容身份。

## 14. 指标和记录

本期不强制 W&B。结构化本地记录是协议要求，终端输出只是人类可读副本。

每个 epoch 至少记录：

- epoch 和 global step
- 实际处理样本数
- train loss
- 当前 learning rate
- validation 是否执行
- `val_CER_all` 和 `val_CER_feasible`
- validation 不可行样本统计
- best checkpoint 是否更新
- epoch wall time

CER 固定为 micro average：

```text
sum(edit_distance(prediction, target)) / sum(target_length)
```

manifest 已禁止空 target，因此分母必须大于零。checkpoint 只在新的
`val_CER_all` 严格小于历史最佳值时替换 `best.pt`；相等时保留较早 checkpoint。

run metadata 记录：

- git commit 和 dirty 状态
- Python、PyTorch、Transformers、PEFT、TorchVision 和 Albumentations 版本
- 操作系统和设备
- protocol、config、manifest 和 vocabulary hashes
- train/val/test 样本数
- 各 split 的 target 长度和 required frames 分布摘要
- 可训练参数数量和名称前缀摘要

## 15. 失败处理

以下情况必须在训练前失败：

- 配置类型或范围无效。
- `start_eval > max_epochs`。
- manifest schema、路径、split 或 group 隔离无效。
- 词表无效或出现 OOV。
- train 存在 CTC 不可行样本。
- backbone revision 未固定。
- patch grid 与 backbone config 不兼容。
- 输出目录不可创建。
- resume metadata 不匹配。

以下情况必须在训练过程中立即失败：

- loss 或 metric 非有限。
- batch 的 target length 与拼接 target 不一致。
- 模型输出时间步与输入契约不一致。
- checkpoint 原子写入失败。

错误信息必须给出具体字段、样本 id 或 checkpoint 字段，不能只返回底层
reshape、`KeyError` 或 `FileNotFoundError`。

## 16. README 和公共行为

README 必须与 v2 行为一致：

- 只描述实际执行的 train、validation 和 test 指标。
- 删除 `engine_test` 会打印样本对的错误陈述，除非实现确实提供该行为。
- 删除无 `__main__` 支持的 `python augments.py` 示例。
- 只展示规范的 patch grid CLI 语法，并说明过渡别名。
- 说明全量 CER 与可行子集 CER 的区别。
- 说明 v2 使用 closed-corpus 词表及其 manifest 身份。
- 说明 legacy checkpoint 不能直接 resume 到 v2。

CLI `--help`、README 和配置校验错误必须使用相同的字段名和允许值。

## 17. 测试设计

测试不得依赖 Hugging Face 网络、私有数据或 CUDA。模型测试使用本地 dummy
backbone，公开与真实 backbone 相同的最小 config 和输出接口。

### 17.1 配置测试

- `start_eval=1` 和 `start_eval=max_epochs` 通过。
- `start_eval<1` 和 `start_eval>max_epochs` 失败。
- 非法 method、patch grid、batch size、workers、learning rate 和 patience 失败。
- `linear_prob` 被规范化为 `linear_probe`。
- 新旧 patch 参数同时出现时失败。
- canonical config 的 Adam 参数和 hash 稳定。

### 17.2 Manifest 和词表测试

- manifest hash 不受 JSON 键顺序和 sample 输入顺序影响。
- 重复 sample、跨 split group、路径逃逸、内容 hash 不匹配、缺失文件和空
  split 失败。
- vocabulary hash 稳定。
- 重复 token、blank 冲突、manifest hash 不匹配和 OOV 失败。

### 17.3 CTC 测试

- `[1, 2]` 需要 2 帧。
- `[1, 1]` 需要 3 帧。
- `[1, 1, 2, 2]` 需要 6 帧。
- train 不可行样本在模型加载前失败。
- val/test 不可行样本保留并进入分层统计。
- 通过预检的 batch 不产生被 `zero_infinity` 隐藏的零梯度。

### 17.4 数据管线测试

- 变长 target collate 生成正确的拼接 tensor 和 lengths。
- 最后一个小 batch 生成正确 input lengths。
- 评估按 target lengths 切片，不依赖 blank 哨兵。
- 超长 target 不导致其他样本被全局 padding。

### 17.5 模型契约测试

- linear probe 和 LoRA 对相同 patch grid 产生相同时间维。
- 两种方法使用同一 exact-grid 路径。
- linear probe 只冻结 backbone。
- LoRA 只有 adapter 和 task head 可训练。
- 非匹配 token count 产生描述性错误。
- 不同 backbone image size 不依赖 `64` 或 `1024`。

### 17.6 Checkpoint 测试

- checkpoint 包含完整 schema。
- 不读取原数据目录也能恢复 id 到 token 的映射。
- `best` 和 `last` 不互相覆盖。
- 同配置两次运行生成不同 run 目录。
- 相同 CPU 环境下，epoch 边界 resume 与不中断运行的下一 epoch tensor 完全
  一致。
- protocol、vocabulary、manifest、revision 或 patch grid 不匹配时拒绝 resume。
- v2 loader 无条件拒绝 legacy state dict。
- summary 中的 checkpoint 文件 hash 与实际文件一致。

### 17.7 CPU 集成测试

使用少量临时图片、固定 manifest、固定词表和 dummy backbone：

1. 完成一次预检。
2. 训练至少一个 epoch。
3. 写入 `last` 和 `best`。
4. 从 `best` 执行 test。
5. 从 `last` 恢复并继续一个 epoch。
6. 验证 run metadata、metrics 和 summary 完整。

## 18. 验收标准

实现只有在以下条件全部满足时才能声明 v2 完成：

1. train 中不存在被静默忽略的 CTC 不可行样本。
2. 相同 manifest、词表和配置产生相同的三个 identity hash。
3. train/val/test 的 group 集合两两不相交。
4. checkpoint 在不扫描原始 target 文件的情况下能够解码预测 id。
5. `start_eval > max_epochs` 在加载模型前失败。
6. 相同配置的两次运行不覆盖任何文件。
7. linear probe 和 LoRA 共享输入几何契约。
8. v2 checkpoint 能在 epoch 边界恢复训练。
9. v2 loader 明确拒绝 legacy checkpoint。
10. 全量 CER、可行子集 CER 和容量统计同时落盘。
11. README、CLI help 和实际参数语义一致。
12. 所有新增测试在 CPU、无网络环境通过。

## 19. 后续阶段

### 19.1 性能阶段

可信基线完成后，单独设计性能协议。候选项包括：

- AMP 与 GradScaler。
- `pin_memory` 和 non-blocking transfer。
- Windows/Linux 分别选择 worker、persistent workers 和 prefetch。
- input lengths 分配与 host/device 传输优化。
- 避免每个 batch 显式创建 LSTM 零状态。

每项必须报告吞吐、峰值显存、训练 loss 数值稳定性和固定验证集 CER parity。
没有测量时不承诺加速倍数。

### 19.2 优化器阶段

AdamW、WSD、warmup、cosine 和梯度裁剪使用独立协议版本。比较时固定 split、
词表、初始化、样本数、batch size 和随机种子。不得把 `full_page_omr`
参数直接复制后视为 staff-level 默认值。

### 19.3 模型消融阶段

行轴聚合至少比较：

- mean pooling
- flatten rows 后 projection
- learned attention pooling

比较使用相同 backbone、输入尺寸、训练样本数、优化协议和解码方式。主要指标
为全量 CER，同时报告参数量、吞吐和显存。只有重复运行显示稳定收益后才改变
默认结构。

## 20. 与 full_page_omr 的关系

本设计借用 `full_page_omr` 已验证的工程原则：

- 配置先校验。
- 数据和词表具有稳定身份。
- checkpoint 保存恢复所需状态。
- 运行协议和指标结构化落盘。
- resume 前验证兼容性。
- 测试覆盖数据、模型、优化和 checkpoint 契约。

本期不直接导入 `experiments/full_page_omr` 的内部模块。两种任务的数据形态、
损失函数和模型 head 不同，强行共用内部实现会制造跨实验耦合。只有当 v2
行为稳定且出现第二个完全相同的 helper 需求时，再考虑把原子写入、hash
规范化或通用配置校验提取到共享模块。
