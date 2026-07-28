# Staff-level OMR 可信训练协议 v2 设计

日期：2026-07-25

状态：已复审，待实现

最后复审：2026-07-29

目标协议：`staff_omr_v2`

## 1. 背景

`experiments/staff_level_omr` 已具备可运行的 MuSViT + BiLSTM + CTC
训练原型，支持 linear probing 和 LoRA。当前实现缺少稳定的数据、输入、
checkpoint 和恢复契约，部分错误会在训练过程中被静默掩盖。

本设计是训练地基重建，不是性能或模型效果优化。本期不以降低 CER 或提高吞吐
为目标；吞吐优化、优化器调整和模型结构消融留到可信基线建立之后，避免同时
改变多个变量而无法解释 CER 变化。

## 2. 决策

本期建立一个版本化的 staff-level OMR 训练协议，满足以下条件：

1. manifest 中每个 train 样本要么参与训练并产生合法 CTC 梯度，要么被配置
   显式排除并留下审计记录，否则在加载模型前触发明确错误。
2. 数据划分、词表、增强、输入几何和基础模型具有稳定身份。
3. run artifact bundle 配合同内容的 `data_path` 和 approved base
   revision/weight identity
   能恢复训练；checkpoint 单文件包含既有预测 id 到符号的映射，但不伪装成
   包含冻结 base 权重的自包含推理模型。
4. 同一配置的重复运行不会覆盖彼此的产物。
5. 评估结果能够说明多少样本受输出长度上限影响。
6. 关键契约由不依赖私有数据、网络和 GPU 的测试覆盖。

## 3. 本期范围

本期包含：

- 单一、经过校验的训练配置。
- 从现有平铺目录生成固定 split manifest 和版本化词表的最小准备命令。
- CTC 可行性预检和分层评估。
- 变长 target 数据管线。
- 显式、版本化的增强和输入几何契约。
- epoch resume 状态完整、原子、且不重复冻结 base 的 checkpoint。
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
- 自动猜测文件名中的 `group_id` 语义。
- 通用数据迁移框架；本期准备命令只支持当前成对的 staff image/target 目录。
- GPU、跨依赖版本或跨设备的完整训练 bitwise resume 保证。
- 把 metadata 不完整的 legacy checkpoint 转换为 v2。

`num_workers`、`max_epochs`、`start_eval` 和 `patience` 会进入统一配置，
目的是消除硬编码并支持校验，不在本期宣称性能收益。

这不是几处“低风险优化”，而是跨数据、模型、训练和产物边界的协议升级。
实现应分两次可审查提交，但不能制造一个短命的中间 checkpoint 协议：

1. 第一片只交付内部 foundation primitives 及其单元测试：dataset bundle、
   配置规范化、CTC policy/排除文件、词表、变长 collate 和分层指标函数。
   `prepare-data` 可以作为独立工具公开；训练入口仍保持 legacy，不得写
   `staff_omr_v2` run、指标或 checkpoint。
2. 第二片原子切换训练 runtime：接入派生输入几何、确定性样本顺序与增强、
   run 目录、完整 checkpoint/resume/finalization、CLI、README 和集成测试，
   同时删除可执行的 v1 训练路径。

因此不存在 `staff_omr_v2_slice1`、临时 checkpoint schema 或一次性迁移器。
第二片在 §18 全部通过前不能自称 `staff_omr_v2` 完成。

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
| `64` 和 `1024` 硬编码 | 输入契约错误 | 从 backbone config 推导；输入几何作为显式协议字段 |
| 两套 `shape_patches` 语法 | CLI 双重事实来源 | 统一公共参数并保留过渡别名 |
| README 与代码不一致 | 用户可见行为错误 | 文档由新契约校正 |
| epoch、patience、workers 硬编码 | 配置不可审计 | 纳入单一配置并验证 |
| manifest 和词表没有生产入口 | 协议无法用于真实数据 | 提供一次性、确定性的 `prepare-data` 命令 |
| 增强未进入配置 hash | 训练身份不完整 | 有序增强契约进入 training contract |
| 增强 RNG 绑定 worker seed | worker 数会改变训练样本 | 样本/epoch 派生增强 seed，worker 不参与语义 |
| 输入几何自由字段却被 method 唯一决定 | 默认 linear probe 会撞组合校验 | v2 由 method 派生并拒绝显式覆盖 |
| 大型排除列表内联配置 | 配置膨胀且难操作 | 使用绑定 manifest/宽度的规范化排除文件 |
| base 权重未进入 registry identity | trainable-only checkpoint 依赖不完整 | 固定权重文件 size、LFS SHA-256 和 pointer oid |
| 每个 checkpoint 嵌入 manifest | 重复 I/O 且无恢复价值 | run 目录保存副本；checkpoint 只保存路径和 hash |

### 4.2 接受问题，调整处理方式

manifest 只描述数据集成员和 group-disjoint split，不描述某个模型时间轴是否
容得下样本。CTC 可行性依赖 `patch_cols`，因此不能把排除结果写回 manifest，
否则同一数据集在不同宽度下会获得不同身份。

训练样本不再由 `filter_max_len` 隐式丢弃。默认
`train_infeasible_policy = fail`；只有显式使用 `exclude_listed` 并提供
规范化排除文件时，才允许排除。排除文件的 canonical hash 和样本数进入
training contract，预检输出数量、比例和 target/required-frames 分布。
manifest、词表和评估集合保持不变。

验证和测试 split 中的不可行样本保留。它们反映模型时间轴容量的真实限制，
但必须同时报告全量 CER、可行子集 CER、不可行样本数量和比例。跨 patch
宽度比较只适用于 v2 的 LoRA/exact-grid；以全量 CER 为主指标，并同时展示
容量统计。

词表不应在每次训练时从 test split 临时 fit。对于 closed-vocabulary OMR，
词表可以来自完整语料，但必须由准备命令生成独立、版本化的词表文件，并在
协议中声明其来源。运行时不得通过读取当前 test 内容来改变输出 head。

linear probe 不强行切换到 exact-grid。冻结 backbone 时，原生输入网格和
原生位置编码更接近 MuSViT 发布的 zero-shot 用法；位置编码插值可能带来无法
由冻结 backbone 吸收的分布变化。v2 为每种 method 固定一个保守几何并把
派生结果显式写入 contract，但不把一个已被 method 唯一决定的值伪装成用户
配置。跨几何比较留作一次性受控实验和后续协议。

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
- 不采用两个彼此独立的 manifest/vocabulary 生成命令。二者共享同一语料扫描
  和 manifest hash，一个原子 `prepare-data` 命令能减少半成品和错配状态。
- `val_CTC_loss_feasible` 值得记录，但只用于诊断；它不改变以全量 CER 选择
  checkpoint 的规则。
- 梯度裁剪不能修复已经出现的非有限 CTC loss。v2 仍以 fail-fast 处理非有限
  loss；是否裁剪梯度必须在后续实验中单独验证。
- 不把 metadata cache 设为可信基线默认。size/mtime 可在内容变化时保持不变，
  与“稳定内容身份”目标冲突；v2 默认全量 image hash，cache 只能显式降级。
- 不因“同一 run 已经校验过”就在 resume 时沿用
  `data_verification=content_verified`。数据文件可在两次进程之间被替换且
  保持 size/mtime；除非数据存储本身不可变或按内容寻址，每次 resume 仍需按
  所选验证模式重新验证。

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

实现 v2 时删除可执行的 v1 训练分支，不在同一运行时保留两套隐式行为。
legacy 行为只存在于 Git 历史或旧 release 中；“兼容”仅指下述 CLI 参数别名，
不包括训练语义兼容。

### 5.2 CLI 兼容

规范入口为：

```text
uv run musvit staff-level-omr prepare-data ...
uv run musvit staff-level-omr train ...
```

现有省略 `train` 的 `musvit staff-level-omr ...` 在一个兼容周期内作为
train alias；help 和运行日志显示规范入口。直接模块入口与规范 `train`
入口调用同一个配置构造和校验函数。公共 long option 统一使用现有风格的
`--snake_case`；README、Fire 和 argparse 不再展示第二种拼写。

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
不猜测优先级。

### 5.3 结果兼容

v2 固定 split、词表、增强、输入几何和评估语义，因此旧 CER 不得与 v2 CER
放入同一结果序列。报告必须展示协议版本、manifest hash、vocabulary hash、
training contract hash 和 input contract。旧结果可作为历史参考，但不能作为
同协议回归基线。

## 6. 单一训练配置

引入一个不可变、经过验证的 `StaffOMRConfig`。CLI、直接 Python 调用和
checkpoint 恢复都使用这一结构。配置分为训练语义和运行环境两层，避免把
`num_workers`、输出路径之类的运维字段误当成模型身份。此前 worker seed 会
改变增强内容的问题由 §12.2 从根上解除，而不是把 worker 数塞进训练身份。

配置至少包含：

| 字段 | 约束 |
|---|---|
| `experiment_name` | 1-64 个 ASCII `[A-Za-z0-9._-]`；非 `.`/`..`/Windows 设备名 |
| `data_path` | 存在且可读 |
| `dataset_bundle_path` | 存在，且包含相互绑定的 manifest、vocabulary 和 verification index |
| `model_name` | 受支持的 MuSViT alias |
| `model_revision` | Hugging Face 不可变 commit SHA |
| `method` | `linear_probe` 或 `lora` |
| `patch_rows` | 正整数 |
| `patch_cols` | 正整数 |
| `augmentation_profile` | `staff_omr_train_v1` 或 `none` |
| `train_infeasible_policy` | `fail` 或 `exclude_listed` |
| `train_exclusions_path` | `exclude_listed` 时必填的规范 JSON 路径；其他 policy 时必须为空 |
| `batch_size` | 正整数 |
| `num_workers` | 非负整数 |
| `learning_rate` | 有限正数 |
| `max_epochs` | 正整数 |
| `start_eval` | `1 <= start_eval <= max_epochs` |
| `patience` | 正整数 |
| `seed` | 非负整数 |
| `output_root` | 可创建目录 |
| `device` | 明确的 `cpu`、`cuda` 或经过记录的 `auto` |
| `verify_image_hashes` | `cached` 或 `always` |

`experiment_name`、`data_path` 和 `dataset_bundle_path` 没有默认值。
`dataset_id` 从 bundle 中读取，不要求调用者重复填写。每个 model alias 必须
带一个经过人工审核、不可变的 `default_approved_revision`；用户不传 revision
时使用该 SHA。以下选项互斥：

```text
--model_revision <immutable-commit-sha>
--resolve_model_revision
```

显式 SHA 必须属于 alias 的 approved revision registry。后一种方式只在新运行
开始前解析 alias 的上游 ref；只有解析到的 SHA 及其机器可验证 evidence 已在
registry 中时才接受，否则要求先审核并更新 alias，绝不自动信任新 revision。
最终 SHA 写入配置后再加载权重；resume 只使用 checkpoint 中的 SHA，不重新
解析浮动 ref。离线运行直接使用 alias 默认或显式 approved SHA。

其余默认值保持当前公开训练行为，并补上 v2 契约字段：

```text
model_name = musvit
model_revision = alias.default_approved_revision
method = lora
patch_rows = 8
patch_cols = 64
augmentation_profile = staff_omr_train_v1
train_infeasible_policy = fail
train_exclusions_path = null
batch_size = 8
num_workers = 6
learning_rate = 3e-4
max_epochs = 1000
start_eval = 20
patience = 30
seed = 7
output_root = experiments/staff_level_omr/runs
device = cuda
verify_image_hashes = always
```

规范化配置由 `method` 唯一派生：

```text
linear_probe -> input_geometry = native_pad
lora         -> input_geometry = exact_grid
```

`input_geometry` 写入规范化配置、`training_contract` 和 `input_contract`，
但 v2 CLI 和公开 Python 构造器都不接受该输入；显式传
`--input_geometry` 必须报错并说明跨几何实验属于 §19.2。这样最自然的
`--method linear_probe` 调用会得到合法的 `native_pad`，不要求用户补一个
由 method 已经决定的参数。

准备好 bundle 后，最小新运行入口必须可直接执行：

```text
uv run musvit staff-level-omr train --experiment_name catedrales-lora --data_path <directory> --dataset_bundle_path <dataset-bundle-directory>
```

除 revision 解析、registry 中小型 config/evidence 的读取以及由此派生的
backbone metadata 校验外，配置和数据预检必须在下载/加载实际 base 权重、
创建 CUDA context、启动 DataLoader workers 之前完成。

配置还固定记录当前优化协议：

```text
optimizer = torch.optim.Adam
betas = [0.9, 0.999]
eps = 1e-8
weight_decay = 0.0
scheduler = none
```

learning rate 使用配置值并保持恒定。v2 不引入 scheduler。
optimizer 只接收 `requires_grad=true` 的参数，并按规范化参数名排序构建单一
param group；名称使用完整 model state-dict key，按 UTF-8 字节序升序，冻结
backbone 参数不得进入 optimizer param group。

配置解析后生成两个 canonical JSON：

1. `training_contract`：包含 protocol、manifest/vocabulary hash、基础模型
   id/revision/weight SHA-256、method、输入几何、patch grid、预处理、完整
   增强契约、CTC
   排除策略、排除文件 canonical hash/数量、task-head schema、CTC loss、
   batch size、seed、`staff_omr_sample_epoch_sha256_v1` 播种协议、样本顺序协议、
   优化器、learning rate、LoRA 参数、`start_eval` 和 `patience`。其
   SHA-256 是实验语义身份。
2. `launch_config`：包含完整用户配置和实际运行字段，包括 `max_epochs`、
   `num_workers`、device、输入文件路径及 image hash 验证模式。每次新建或
   恢复调用均保存一份及其 SHA-256，用于审计，不作为一刀切的 resume
   拒绝条件。

`max_epochs` 是预算上限而不是每步训练语义，故不进入 `training_contract`；
只允许在 resume 时增大。`experiment_name`、路径和 worker 数也不进入
`training_contract`。任何允许变化的字段都必须记录在 resume event 中，不能
静默覆盖最初配置。

所有 canonical JSON hash 使用同一序列化规则：UTF-8、对象键按字典序、
`ensure_ascii=false`、分隔符 `,` 和 `:` 周围无空格、无结尾换行。manifest 的
samples 在序列化前按 `sample_id` 排序，路径统一为 `/` 分隔的相对路径；
vocabulary 的 tokens 数组顺序具有语义，不排序。

## 7. 数据和 split manifest

### 7.1 Dataset bundle 和 manifest schema

训练只接收一个 dataset bundle。bundle 根文件为：

```json
{
  "schema_version": "staff_omr_dataset_bundle_v1",
  "dataset_id": "catedrales",
  "prepare_protocol_version": "staff_omr_prepare_v1",
  "manifest_file": "split_manifest.json",
  "manifest_sha256": "<64 lowercase hex chars>",
  "vocabulary_file": "vocabulary.json",
  "vocabulary_sha256": "<64 lowercase hex chars>",
  "image_verification_index_file": "image_verification_index.json",
  "generation_contract": {
    "pair_rule": "replace_final_suffix_v1",
    "image_suffix": "_region.png",
    "target_suffix": "_gt.txt",
    "group_regex": "(?P<group_id>...)",
    "split_weights": [8, 1, 1],
    "seed": 7,
    "split_algorithm": "sha256_group_largest_remainder_v1",
    "target_parser": "utf8_unicode_whitespace_v1",
    "token_sort": "utf8_bytes_ascending_v1"
  }
}
```

CLI ratio 先按十进制精确值转成最简正整数 weights，不在 identity JSON 中保存
平台相关 float。bundle 中三个成员文件名固定为不含目录分隔符的上述名称。
`dataset_bundle_sha256` 是 `bundle.json` canonical 内容的 SHA-256；verification
index 不进入该 hash。bundle hash 是生成 provenance；实验主身份仍由最终
manifest、vocabulary 和 training contract 三个 hash 决定。若两种生成过程
逐字节产生同一 manifest/vocabulary，它们可以共享训练身份，但 provenance
必须分别保留。

v2 不在训练启动时调用 `train_test_split`。bundle 内需要一个持久化 manifest：

```json
{
  "schema_version": "staff_omr_split_v1",
  "dataset_id": "catedrales",
  "group_semantics": "user_declared_regex",
  "samples": [
    {
      "sample_id": "score01-page03-staff02",
      "group_id": "score01-page03",
      "image_path": "score01-page03-staff02_region.png",
      "image_size_bytes": 123456,
      "image_sha256": "<64 lowercase hex chars>",
      "target_path": "score01-page03-staff02_gt.txt",
      "target_sha256": "<64 lowercase hex chars>",
      "split": "train"
    }
  ]
}
```

路径相对 `data_path` 解析。`sample_id` 和 `group_id` 必须非空且稳定。

只读 image verification index schema 为：

```json
{
  "schema_version": "staff_omr_image_verification_v1",
  "source_manifest_sha256": "<64 lowercase hex chars>",
  "entries": [
    {
      "image_path": "score01-page03-staff02_region.png",
      "image_size_bytes": 123456,
      "image_mtime_ns": 1234567890000000000,
      "image_sha256": "<64 lowercase hex chars>"
    }
  ]
}
```

entries 按 `image_path` 排序。mtime 只用于 cache 命中，不进入 manifest 或
dataset identity。

### 7.2 Manifest 校验

启动前验证：

1. bundle、manifest、vocabulary 和 verification index 的 schema version 受支持。
2. bundle 记录的 manifest/vocabulary canonical hash 与文件实际内容相等。
3. `bundle.dataset_id == manifest.dataset_id == vocabulary.dataset_id`。
4. vocabulary 和 verification index 的 `source_manifest_sha256` 均等于 bundle
   记录的 manifest hash。
5. bundle 的固定成员文件名不含 `/`、`\` 或 `..`，解析后仍位于 bundle 内。
6. `sample_id` 唯一。
7. 每个 `group_id` 只出现在一个 split。
8. split 只允许 `train`、`val`、`test`，且三个 split 均非空。
9. image 和 target 文件存在且可读。
10. target 内容 SHA-256 与 manifest 一致。
11. image 大小等于 `image_size_bytes`，并按 7.4 的模式校验内容 SHA-256。
12. verification index 对 manifest 中每个 image 恰有一项，path、size 和 hash
    与 manifest 相等，不得含多余 entry。
13. target 解析后非空。
14. 同一文件不能被多个 sample 引用。
15. 所有数据路径解析后仍位于 `data_path` 内。

manifest 以 `sample_id` 排序后规范化并计算 SHA-256。训练不修改 manifest。

### 7.3 数据准备命令

v2 必须提供一个最小、一次性的准备入口，使现有平铺目录不需要额外脚本：

```text
uv run musvit staff-level-omr prepare-data --data_path <directory> --dataset_id <stable-id> --group_regex '<regex-with-named-group-group_id>' --split_ratios 0.8 0.1 0.1 --seed 7 --out <dataset-bundle-directory>
```

该命令执行一遍确定性流水线，在一个 bundle 中生成 `bundle.json`、
`split_manifest.json`、`vocabulary.json` 和 `image_verification_index.json`，不提供
两个可独立执行而产生错配的子命令。`bundle.json` 固定记录 bundle schema、
dataset id、prepare protocol version、manifest hash 和 vocabulary hash；
并保存 regex、最简 integer weights、seed、split algorithm 和 target parser
版本。
verification index 不参与数据集身份 hash，训练只读，不在并发运行中改写。

1. 分别递归扫描 `*_region.png` 和 `*_gt.txt`，按规范化相对路径排序。配对只
   允许把 image 路径末尾最后一次 `_region.png` 替换为 `_gt.txt`；反向规则
   同理。孤立 image、孤立 target 和大小写规范化后冲突的路径均失败。
2. `group_regex` 对完整的 `/` 分隔 image 相对路径执行 full match，且必须定义
   非空命名捕获组 `group_id`。不匹配项全部写入错误报告，终端展示总数和前
   20 项；工具不猜测 score/page 语义。
3. `sample_id` 是去掉 `_region.png` 后的规范化相对路径；冲突即失败。
4. 对每个 group 计算
   `SHA256(dataset_id + "\0" + decimal(seed) + "\0" + group_id)`，按
   `(digest, group_id)` 排序。
5. `split_ratios` 必须是三个有限正十进制数；按精确十进制值扩大为整数并除以
   最大公约数，得到最简 `split_weights`。至少需要三个 group。设
   `remaining = group_count - 3`，每个 split 的初值为 1，再对
   `remaining * weight / sum(weights)` 取 floor，并按精确分数余数从大到小
   补齐未分配 group；余数相同时按 `train, val, test` 顺序。排序后的 group
   按最终 count 依次填入 train、val、test。
6. 报告各 split 的 group 数、样本数和 target length 分布。用户若不接受
   group 大小不均造成的样本比例偏差，应更换显式 ratio/seed 后重新生成，
   工具不把同一 group 拆开追求样本数平衡。
7. 解析所有 target，计算内容 hash，从全语料 token 集合生成词表；同时计算
   image 大小和内容 hash。
8. 全部文件先写入同级临时目录，完整通过 schema 和交叉校验后，再把目录
   rename 为最终 `--out`。最终目录已存在时失败；任一步失败都不暴露半个
   bundle。

manifest 不包含 `patch_rows`、`patch_cols`、CTC 可行性或训练排除项。给定相同
输入内容、dataset id、regex、最简 weights、seed 和工具协议版本，`bundle.json`、
manifest 和 vocabulary 必须逐字节稳定；含本地 mtime 的 verification index 是
可重建的
运行加速，不要求逐字节稳定。

### 7.4 Image 内容验证成本

`prepare-data` 总是计算一次完整 image SHA-256，并写入 bundle 内、由 manifest
hash 绑定的只读 verification index。训练默认
`verify_image_hashes=always`：

- target 始终重新计算内容 hash，因为它体积小且决定词表和 CTC 可行性。
- image 始终检查存在性和 `image_size_bytes`。
- `verify_image_hashes=always` 忽略 index，对全部 image 重新计算内容 hash。
- 只有调用者显式选择 `verify_image_hashes=cached` 时，index 中 manifest
  hash、相对路径、size 和 mtime 均匹配的 image 才跳过重算；缺失或任一字段
  变化时重新计算该文件 hash。

每次进程启动（包括 resume）都重新执行上述验证并记录 index 命中数、重算数
和验证模式。`always` 的 run 标记为
`data_verification=content_verified`；`cached` 是显式的性能与检测强度降级，
标记为 `data_verification=cached_metadata`，不得把它描述为对保持 size/mtime
修改的内容证明，也不得用作发布 v2 可信基线的验收 run。先前进程的成功记录
只能证明当时读取的内容，不能替代 resume 时的验证。

## 8. 词表契约

词表使用独立 JSON：

```json
{
  "schema_version": "staff_omr_vocab_v1",
  "dataset_id": "catedrales",
  "vocabulary_scope": "closed_corpus",
  "source_manifest_sha256": "<64 lowercase hex chars>",
  "target_parser": "utf8_unicode_whitespace_v1",
  "token_sort": "utf8_bytes_ascending_v1",
  "unicode_normalization": "none",
  "blank_id": 0,
  "tokens": ["token-a", "token-b"]
}
```

`tokens[0]` 映射到 id 1，blank 固定为 id 0。token 不得为空或重复。
模型输出类别数固定为：

```text
num_classes = len(tokens) + 1
token_id(tokens[i]) = i + 1
blank_id = 0
```

运行前验证所有 split 的 token 都存在于词表。出现 OOV 时列出 split、样本 id
和 token 计数，然后终止。运行时不重新 fit `LabelEncoder`。

v2 首版只接受 `vocabulary_scope = closed_corpus`。词表由 7.3 的
`prepare-data` 命令生成：target 按 UTF-8 解码，以 Python `str.split()` 语义
按 Unicode whitespace 分词，不做 Unicode normalization，token 按 UTF-8
字节序升序排列。该 parser/sort 规则的版本写入 vocabulary schema，不能依赖
locale。
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

预检先计算所有满足下式的训练样本：

```text
minimum_frames(target) > patch_cols
```

处置由配置决定：

- `fail`：集合非空即在加载 backbone 前终止。失败 run 写出可直接复用的
  `train_exclusions.candidate.json`，错误信息展示其路径。
- `exclude_listed`：`train_exclusions_path` 的 canonical sample id 集合必须
  与预检得到的不可行训练集合完全相等。漏列、重复、列入可行样本或列入
  val/test 样本均失败。通过后只从本次训练 DataLoader 排除这些 id，不修改
  manifest。

排除文件是唯一受支持的列表输入，不再同时维护 inline/file 两种形态：

```json
{
  "schema_version": "staff_omr_train_exclusions_v1",
  "source_manifest_sha256": "<64 lowercase hex chars>",
  "patch_cols": 64,
  "required_frames_algorithm": "ctc_minimum_frames_v1",
  "sample_ids": [
    "score01-page03-staff02"
  ]
}
```

`sample_ids` 按 UTF-8 字节序排序且不得重复；manifest hash、`patch_cols` 和算法
必须与本次运行完全相等。文件按 §6 的 canonical JSON 规则计算 SHA-256。
`exclude_listed` 时 training contract 保存 policy、canonical hash 和 count，
launch config 另存源路径；run 内复制为 `train_exclusions.json`，resume 只读
该副本。`fail` policy 的 contract 固定保存 exclusion hash 为 JSON `null`、
count 为 0；生成的 candidate 不会反向改变本次失败 run 的身份。candidate
使用相同 schema，因此用户审阅后可以直接作为下一次运行的
`train_exclusions_path`，不需要复制上万条 CLI 参数。

应用 policy 后必须满足 `retained_train_samples > 0`；全量 train 都不可行时
在创建 DataLoader 前失败，不能把空集合留给 shuffle sampler 或 loss 聚合。

无论哪种策略，preflight 和 run metadata 都记录：

- 不可行样本数和占 train 的比例。
- 最大 required frames。
- 每个失败样本的 `sample_id`、target length、adjacent repeats、required frames
  和 available frames。
- 排除样本与保留样本各自的 target length、adjacent repeats 和 required
  frames 分布摘要。

训练循环本身不再决定丢弃。CTC loss 契约固定为：

```text
blank = 0
reduction = mean
zero_infinity = false
log_probs_layout = T,N,C
```

所有字段进入 `training_contract`。任何非有限 loss 都作为协议违反立即报错。

### 9.2 Validation 和 test

不可行样本保留，并在 sample metadata 中标记。评估同时输出：

```text
val_CER_all
val_CER_feasible
val_CTC_loss_feasible
val_infeasible_samples
val_infeasible_ratio

test_CER_all
test_CER_feasible
test_infeasible_samples
test_infeasible_ratio
```

`val_CER_all` 是 checkpoint 选择指标。`val_CER_feasible` 用于解释容量影响，
不能替代全量指标。validation 对可行样本额外计算
`val_CTC_loss_feasible`，仅用于诊断，不参与 early stopping 或 checkpoint
选择；不可行样本仍参与 `val_CER_all`，但不送入 CTC loss。test 不计算 loss。

`val_CTC_loss_feasible` 使用 `reduction=none`，将每个样本的 loss 除以其
target length 后对可行样本取算术平均，避免最后一个 batch 改变权重。若某个
split 没有可行样本，其 `CER_feasible` 以及 validation 的 feasible loss 写为
JSON `null`，同时记录 feasible sample count 为 0。

v2 中只有 `lora + exact_grid` 能改变 `patch_cols`。`linear_probe +
native_pad` 要求 `patch_cols == native_cols`，对当前两个 approved MuSViT
alias 均由配置推导为 64；runtime 仍不得写死 `64`。LoRA 跨宽度报告可以共享
manifest、词表和全量评估集合，但若训练排除文件不同，就不是只改变宽度的
纯因果比较。报告必须并列展示各自排除文件 hash 和分布，不得隐藏这一训练
分布差异。linear probe 与 LoRA 的比较同时改变 method 和几何，不能伪装成
单变量宽度消融。

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

DataLoader 语义固定为：

```text
train sampler = sha256_epoch_order_v1 (§12.2)
train shuffle = false  # sampler 已给出完整顺序
val/test order = manifest sample_id order
in_order = true
drop_last = false
persistent_workers = false
```

显式 sampler 与 `shuffle=true` 不得并用。`in_order=true` 保证 worker 完成顺序
不改变 batch 交付顺序；每个 epoch 重新创建 worker，但增强 seed 不再来自
worker。以上语义字段进入 `training_contract`，`num_workers` 本身只进入
launch config，并由跨 worker 的逐字节输入测试证明它不是训练语义。

## 11. 输入和模型契约

### 11.1 输入几何

`input_geometry` 是协议派生字段，而不是用户自由配置，也不能散落在模型内部
偷偷分支。配置规范化只生成两个经过命名的组合：

| method | input_geometry | v2 行为 |
|---|---|---|
| `linear_probe` | `native_pad` | 冻结 backbone，保持预训练原生位置网格 |
| `lora` | `exact_grid` | adapter 可训练，按目标网格插值位置编码 |

公开入口显式提供 `input_geometry` 一律报错。字段仍分开保存到 contract，
是为了让产物自描述，并让未来新协议能开放受控组合而不改变名称含义。

`native_pad`：

```text
native_rows = backbone_image_height / backbone_patch_height
native_cols = backbone_image_width  / backbone_patch_width
content_height = patch_rows * backbone_patch_height
content_width  = patch_cols * backbone_patch_width
```

要求 `patch_rows <= native_rows` 且 `patch_cols == native_cols`。image 先 resize
到 content size，再只在底部用 RGB `[255, 255, 255]` pad 到 backbone 原生 image
height；不做水平 pad，不插值位置编码。移除 prefix token 后先按
`native_rows x native_cols` reshape，再只取顶部 `patch_rows`。这些规则复现
linear probe 的原生位置网格意图，但尺寸全部来自 backbone config。

`exact_grid`：

```text
height = patch_rows * backbone_patch_height
width  = patch_cols * backbone_patch_width
```

image 直接 resize 到该尺寸，不 crop 或 pad，并在 `ViTModel` 调用中显式传
`interpolate_pos_encoding=true`。位置编码插值契约固定为
`mode=bicubic, align_corners=false, antialias=false`，对应审核过的 Transformers
`ViTEmbeddings.interpolate_pos_encoding`；移除 prefix token 后按
`patch_rows x patch_cols` reshape。

两种几何都不保持原宽高比，resize 使用 bilinear interpolation 并明确启用
antialias。所有尺寸、图像插值、padding、位置编码插值和切片字段写入
`input_contract`，不能依赖 TorchVision 的隐式默认值。bilinear 与 antialias
是 v2 为确定性作出的项目协议选择；官方示例没有显式固定这两个参数，不能把
它们描述成上游保证。preflight 还要把实际 position embedding 输入送入
审核过的 bicubic reference，并与已安装 Transformers 的插值结果比较；结果
不一致即拒绝运行，不能只靠 source string 或版本号猜行为。

### 11.2 增强与 tensor 预处理

执行顺序固定为：

1. image 解码为三通道 RGB。
2. train 按 `augmentation_contract` 增强；val/test 不增强。
3. 执行 11.1 的 resize/pad。
4. 按 `ToTensor` 语义转为 float tensor 并缩放到 `[0, 1]`。
5. 不做 mean/std normalization。

`staff_omr_train_v1` 必须序列化为有序、声明式结构，至少完整展开以下当前策略
及所有行为参数；实现不能依赖 Albumentations 默认值：

```text
Compose(p=1.0)
  OneOf(p=0.6)
    Morphological(scale=[2,2], operation=dilation, p=1.0)
    Morphological(scale=[2,2], operation=erosion, p=1.0)
  Sharpen(alpha=[0.2,0.5], lightness=[0.5,1.0],
          method=kernel, kernel_size=5, sigma=1.0, p=0.25)
  Rotate(limit=[-3,3], interpolation=linear, border_mode=replicate,
         rotate_method=largest_box, crop_border=false,
         mask_interpolation=nearest, fill=0, fill_mask=0, p=0.5)
  GaussNoise(std_range=[0.01,0.15], mean_range=[0,0],
             per_channel=false, noise_scale_factor=1, p=0.3)
  ColorJitter(brightness=[0.25,1.75], contrast=[0.25,1.75],
              saturation=[0.25,1.75], hue=[-0.05,0.05], p=0.75)
  OneOf(p=0.25)
    GaussianBlur(blur_limit=[3,4], sigma_limit=[0.5,3.0], p=1.0)
    MotionBlur(blur_limit=[3,4], allow_shifted=true,
               angle_range=[0,360], direction_range=[-1,1], p=1.0)
  ToGray(num_output_channels=3, method=weighted_average, p=0.1)
```

规范化结构、顺序和 SHA-256 全部进入 `training_contract`。profile 名相同但展开
内容不同必须产生不同 contract hash；库升级改变解析结果时也必须在 preflight
中报出差异。

`[0,1]` 且不 normalization 不是任意选择。MuSViT 官方模型说明当前示例使用
`Resize([1024, 1024])` 后直接 `ToTensor()`，并对 staff/non-page zero-shot
建议 padding 到原生 1024，同时提示位置编码插值可能降低 zero-shot embedding
质量但适用于 fine-tuning：
<https://huggingface.co/PRAIG/musvit/blob/0e91c7b223b4da30f259198c92045d0cb90e3f2e/README.md>。

上段只解释人工决策，runtime 不解析 README 自然语言。v2 alias registry 固定
以下截至 2026-07-29 已复核 evidence：

```text
musvit:
  model_id: PRAIG/musvit
  default_revision: 0e91c7b223b4da30f259198c92045d0cb90e3f2e
  README.md: {git_blob_oid: 8789d81c7e698c92b57746ab5dc090fe893069e2,
              size: 6105}
  config.json: {git_blob_oid: dc2f7bc9bf1aab858aaeefaf1db7858c427f79f2,
                size: 665}
  model.safetensors:
    {git_lfs_pointer_blob_oid: e935a37bc0a6ca051a091a2ba5b5425547f16af9,
     lfs_content_sha256: 109bbaf31d9f2184df1b841579e06d25bc58ed6a42a10dd5f4a5d27d01889db2,
     size: 467638680}
  preprocessor_config.json: absent
  reviewed_input_contract: staff_omr_input_v2

musvit_light:
  model_id: PRAIG/musvit-light
  default_revision: adf40fd3eaf157e20aaa8603ffea06517e467c7f
  README.md: {git_blob_oid: 3155fc2c6beac85bcfd0146de779a65dc872cc8c,
              size: 6144}
  config.json: {git_blob_oid: b9f746f5b2ecdd858c7dfbf4c724ae1de258b1a7,
                size: 663}
  model.safetensors:
    {git_lfs_pointer_blob_oid: 5c03c2357110e0060f3687d8d79dc5230c43019e,
     lfs_content_sha256: f2c278f2762a88bfcc7ee4cf846d8eef2f31c04a73e7775124db69e7afd0528f,
     size: 157546736}
  preprocessor_config.json: absent
  reviewed_input_contract: staff_omr_input_v2
```

registry evidence 来自对应不可变 revision 的官方 Hugging Face tree metadata：
<https://huggingface.co/PRAIG/musvit/tree/0e91c7b223b4da30f259198c92045d0cb90e3f2e>
和
<https://huggingface.co/PRAIG/musvit-light/tree/adf40fd3eaf157e20aaa8603ffea06517e467c7f>。

复核方式不是从文件名或页面文本推断：使用官方 Hub API
`model_info(revision=<commit>, files_metadata=True)` 确认 resolved SHA、size、
Git blob oid 和 LFS metadata；再下载 README/config 原始 bytes，独立按
`SHA1("blob " + decimal(byte_length) + NUL + content_bytes)` 重算普通 Git
blob oid。权重不为复核 pointer 而下载数百 MB；其标准 Git LFS pointer 由
官方 `sha256`/size 重建并用同一 Git blob 公式复算 pointer oid。上述两组
resolved revision、四个普通 blob oid、两个 LFS pointer oid 均于
2026-07-29 匹配。

运行时在把 base 权重交给 Transformers 前，还必须对实际
`model.safetensors` bytes 计算 size 和 SHA-256，并匹配 registry 中的
`lfs_content_sha256`；pointer oid 不能替代内容校验。启动只做可机器判定的
revision、文件存在性、size/hash、processor absence 和 config 字段检查，并把
匹配的完整 evidence record 写入 run metadata 和 checkpoint。不在 registry
的 revision、预期 absent 的 processor 文件出现、权重内容不匹配或 config
字段不兼容都立即失败；接受新上游 revision 必须先人工复核并版本化更新
registry 和 protocol/input contract。

### 11.3 Backbone metadata

v2 的 MuSViT alias 固定使用
`transformers.ViTModel.from_pretrained(model_id, revision=commit_sha,
trust_remote_code=False, add_pooling_layer=False, output_loading_info=True)`，
不用
`AutoModel` 或 `ViTMAEModel`。官方说明中后两者可能保留 MAE masking/shuffle
语义，而下游 spatial token 路径需要完整、有序 patch。

标准 `ViTConfig` 不提供通用 `prefix_token_count` 字段，因此 v2 alias contract
显式固定 `prefix_tokens=1`，对应官方示例跳过的 CLS token。加载后用实际输出
验证总 token 数严格等于 `1 + expected_spatial_tokens`，再移除第一个 token；
不是从 config 猜 prefix 数。

patch size、hidden size 和原始 image size 先从 registry 已验证的
`config.json` 读取以完成无权重 preflight，再与加载后的 backbone config
逐字段校验。alias 配置只额外保存模型 id、固定 revision、loader class 和
`prefix_tokens=1`，不复制可能漂移的维度常量。backbone image height/width
必须分别被 patch height/width 整除，否则所选 v2 几何不受支持并立即失败。
approved raw config 要求 `model_type=vit_mae` 且 architectures 包含
`ViTMAEForPreTraining`；v2 按官方下游示例显式选择 `ViTModel` encoder loader，
不让 raw architecture 自动决定类。同时要求 `num_channels=3`，image/patch
各维和 hidden size 均为正整数；不满足时不尝试兼容。

loading info 要求 missing/mismatched keys 为空；unexpected keys 只允许 approved
MAE checkpoint 中未被 encoder 使用的 `decoder.*`。出现其他 key 即失败，避免
静默保留随机初始化的 backbone 参数。

删除 v2 实现中的 `64x64` 和 `1024` 字面量以及整个 v1 运行分支。backbone
输出移除 prefix token 后必须严格满足所选几何的 token 契约：

```text
exact_grid: spatial_token_count == patch_rows * patch_cols
native_pad: spatial_token_count == native_rows * native_cols
```

不满足时抛出包含模型 id、revision、输入尺寸、patch size 和实际 token 数的
错误。

### 11.4 本期保持的模型结构

“保持不变”必须可机器校验。`task_head_contract` 完整固定为：

```text
schema_version = staff_omr_task_head_v1
spatial_order = rows_then_columns
input_dropout = {p: 0.25, position: before_projection}
projection = {
  in_features: backbone_hidden_size,
  out_features: 256,
  bias: false
}
row_pool = {operation: mean, axis: rows, position: after_projection}
rnn = {
  type: LSTM,
  input_size: 256,
  hidden_size: 256,
  num_layers: 2,
  bias: true,
  batch_first: true,
  dropout: 0.5,
  bidirectional: true,
  proj_size: 0,
  initial_state: zeros_same_dtype_and_device_as_input
}
classifier = {in_features: 512, out_features: num_classes, bias: true}
output = {log_softmax_dim: -1}
decoder = {type: greedy_ctc, collapse_repeats: true, remove_blank_id: 0}
```

执行顺序固定为 spatial tokens → reshape/slice → dropout → projection →
row mean → LSTM → classifier → log-softmax。contract 进入
`training_contract`；参数名、shape、bias 和 state-dict schema 任一变化都必须
改变 contract 或 protocol，不能只改代码。

task head state-dict key 集合固定为：

```text
projection.weight
rnn.(weight_ih|weight_hh|bias_ih|bias_hh)_l{0,1}
rnn.(weight_ih|weight_hh|bias_ih|bias_hh)_l{0,1}_reverse
classifier_ctc.weight
classifier_ctc.bias
```

正向和 reverse 的四类 RNN key 在两层上都必须存在，不能有
`projection.bias`。backbone 与 LoRA adapter 使用各自固定 revision/config
生成的 key，loader 另行做严格 missing/unexpected-key 校验。

LoRA 参数保持当前协议并写入 `training_contract`：

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
2. 必要时解析 model revision，冻结为 commit SHA；读取并校验 registry 中的
   小型 config/evidence，派生 method 对应的输入几何，但不加载权重或创建
   CUDA context。
3. 加载并校验 manifest、词表和 image verification index。
4. 创建 run 目录，复制规范化 bundle/manifest/vocabulary/verification index，
   写入 `status=preflighting` 的初始 `run.json`。
5. 读取 target，完成 OOV、split、增强契约和 CTC 预检；写入 preflight
   结果，并在需要时写出 candidate 或规范化排除文件。预检失败时把 run
   标为 `failed`，但仍不得加载 backbone。
6. 设置 `init_seed`，校验实际 base weight 内容 SHA-256 并加载固定 revision
   的 backbone。
7. 复核已加载 config、官方预处理依据、位置编码 reference、输入和 token
   geometry。
8. 构建 Dataset、DataLoader、模型和优化器。
9. 训练并按配置执行验证。
10. 每个完整 epoch 结束后原子更新 `last`；验证严格改善时原子更新 `best`。
11. 从 `best` checkpoint 执行 test。
12. 写入最终 run summary。

### 12.1 Epoch 和 early stopping

`max_epochs`、`start_eval` 和 `patience` 不再硬编码。

必须满足：

```text
1 <= start_eval <= max_epochs
patience >= 1
```

从 `start_eval` 起每个 epoch 执行一次 validation。`patience` 统计连续多少次
已执行的 validation 没有严格改善 `val_CER_all`；达到阈值后停止。

每个 epoch 是一个明确的事务，顺序不得调整：

1. 完成全部 train batch 并聚合 train 指标。
2. 若到达 `start_eval`，完成 validation 和可行 loss/CER 聚合。
3. 根据本 epoch 指标更新 `best_metric_value`、`best_epoch`、patience、
   `best_updated` 和 stop decision。
4. 构造包含上述更新后状态及完整 `epoch_record` 的 `last.pt` 临时文件，原子
   replace 正式 `last.pt`；这是该 epoch 唯一 commit point。
5. 若 `best_updated=true`，用同一模型和更新后状态原子更新 `best.pt`。
6. 把已提交的 `epoch_record` 追加到 `metrics.jsonl`，再原子更新
   `run.json` sidecar。
7. 若 stop decision 为真，结束循环；否则进入下一 epoch。

`stop_reason` 的优先级固定为：patience 达阈值时
`early_stopping`；否则当前 epoch 达到预算时 `max_epochs`；否则 JSON
`null`。两者同 epoch 发生时按 `early_stopping` 处理，不能靠延长预算绕过。

若进程在步骤 4 前失败，本 epoch 未提交；步骤 4 后失败时，`last.pt` 是权威
状态。resume 必须先用其中的 `epoch_record` 修复滞后的 metrics/run sidecar；
若它同时记录 `best_updated=true` 且 `best.pt` 落后，则从 `last.pt` 修复
同一 state 重新序列化 `checkpoint_role=best` 的 `best.pt`。修复完成后再按
`stop_reason` 决定是否允许进入 `next_epoch`，不能
把 sidecar repair 当作继续训练许可。

`metrics.jsonl` 修复规则固定为：只承认以换行结束且能解析的 canonical JSON
记录；先截断崩溃留下的尾部半行，再要求已有 epoch 从 1 连续递增且不超过
`last.pt.epoch`。正常可修复状态只允许缺少 last 对应的最后一条：若缺失则
追加，若同 epoch 记录逐字段相等则不重复写；更早的 gap、同 epoch 内容冲突
或文件领先于 last 均按损坏失败。这样 checkpoint 仍是 commit point，同时
JSONL 保持 O(n) 总写入而不是每个 epoch 重写历史。

验证指标为 NaN 或 infinity 时立即失败，不能把既有旧 checkpoint 当作本次
训练结果。每个新运行使用空的新 run 目录；resume 只更新原 run。

### 12.2 初始化和 epoch 随机性

`seed_schedule_version` 固定为
`staff_omr_sample_epoch_sha256_v1`。所有 seed 派生先定义同一个无歧义原语：

```text
seed32(label, parts...) =
  int.from_bytes(
    SHA256(UTF8("staff_omr_v2") + NUL
           + UTF8(label) + NUL
           + NUL.join(UTF8(part) for part in parts)).digest()[0:4],
    byteorder="big",
    signed=false)
```

整数 part 使用无前导零的十进制 ASCII；`sample_id` 使用 manifest 中的原始
Unicode 字符串，且路径规则已禁止 NUL。模型初始化使用与 epoch 独立的：

```text
init_seed = seed32("init", decimal(seed))
```

在加载 backbone、创建 LoRA adapter、task head 和 optimizer 之前，用
`init_seed` 重置 Python、NumPy 和 Torch CPU/CUDA RNG。固定 revision 的
backbone 不允许留下未加载而随机初始化的参数；loader 的 missing/mismatched
key 必须通过 alias contract 明确允许，否则失败。base 成功加载后，在创建
trainable modules 前再次重置同一 `init_seed`；LoRA 固定先建 adapter 再建
task head，linear probe 只建 task head。初始化因此不依赖 backbone 构造过程
消耗了多少随机数，同环境、同 contract 的两个新运行在第一个 batch 前具有
相同的全部可训练 state dict。

每个 epoch 开始前：

```text
epoch_seed = seed32("epoch", decimal(seed), decimal(epoch))
worker_base_seed = seed32("worker", decimal(seed), decimal(epoch))
sample_augment_seed =
  seed32("augment", decimal(seed), decimal(epoch), sample_id)

sample_order_key =
  SHA256(UTF8("staff_omr_v2") + NUL + UTF8("order") + NUL
         + UTF8(decimal(seed)) + NUL + UTF8(decimal(epoch)) + NUL
         + UTF8(sample_id))
```

train 保留样本按 `(sample_order_key bytes, UTF8(sample_id))` 升序构成该 epoch
的完整 sampler；sampler item 固定为 `(epoch, manifest_sample_index)`，让
Dataset 不依赖进程本地的可变 `current_epoch`。这就是
`sha256_epoch_order_v1`，不调用 DataLoader `shuffle`。用 `epoch_seed` 重置
Python、NumPy 和 Torch CPU/CUDA RNG，使 dropout 等训练随机性从 epoch 边界
可重建。DataLoader 使用单独、以 `worker_base_seed` 初始化的 generator，
避免 worker base-seed 分配消耗模型训练 RNG。

worker init 可以把 PyTorch worker seed 传给 Python/NumPy 作为非语义卫生措施，
但禁止用它设置增强 pipeline。Dataset 的 `__getitem__` 必须在每次应用
Albumentations 前调用
`Compose.set_random_seed(sample_augment_seed)`；增强之后不得再读取 worker
全局随机流来改变样本。于是同一 `(seed, epoch, sample_id)` 的增强图像和同一
epoch 的 batch 顺序与 shuffle 落位、预取时序及 `num_workers` 无关。
`num_workers=0` 使用完全相同路径，没有单独播种特例。这样 epoch 边界恢复
不依赖上个进程的隐藏 RNG 状态，同时明确改变了 legacy 的播种协议。

### 12.3 Resume 边界

原 run 只允许从其 `checkpoints/last.pt` 恢复；显式 checkpoint 也必须带
`checkpoint_role=last` 并能解析到同一 run bundle。`best.pt` 只用于 test 或
推理，禁止作为 resume 起点，避免回滚并覆盖更新的权威状态。

resume 从 `run.json` 和 `last.pt` 重建原配置；CLI 只开放 `max_epochs`、运行
环境字段及 `--allow_env_drift`，不要求调用者重新拼出整套训练参数。

v2 只承诺完整 epoch 边界的 resume，不承诺 batch 中间恢复。checkpoint 保存
模型、优化器、early-stopping、global step、base seed、seed schedule version
和 `next_epoch`；epoch 顺序、训练 RNG 和逐样本增强 seed 均由这些稳定字段
重建，不序列化各库任意形态的内部 RNG object。

resume 校验分三级。

**硬拒绝，不能 override：**

- protocol/schema version、training contract SHA-256。
- manifest、vocabulary、排除文件、基础模型 revision/weight SHA-256。
- method、input geometry、patch grid、seed、优化器和增强契约。
- checkpoint tensor/schema 与待构建模型不兼容。
- 在调用 `optimizer.load_state_dict()` 前，checkpoint 的有序
  `optimizer_parameter_names` 必须与当前单一 param group 的完整参数名逐项
  相等；不能依赖 PyTorch 按位置给同 shape 参数套用 state。
- Python、PyTorch、TorchVision、Transformers、PEFT、NumPy、Albumentations
  和 OpenCV 的 major.minor 版本不同。

**允许的预算延长：**

- 只有因达到 `max_epochs` 正常结束或尚未终止的 run 才能把 `max_epochs`
  增大，且新值必须大于等于 `next_epoch`。
- 因 early stopping 结束的 run 在相同 `patience` 下是终态，不能靠增加
  `max_epochs` 继续。
- `stop_reason=early_stopping` 修复 sidecar 后进入 finalization，不再训练；
  `stop_reason=max_epochs` 只有预算按上条增大后才继续，否则进入 finalization；
  `stop_reason=null` 才可直接从 `next_epoch` 继续。
- 原值、新值、时间和调用参数写入 append-only resume event；原始
  `launch_config` 不被覆盖。

**环境漂移：**

上述关键库的 patch 版本、其他非关键 package 的任意版本、OS/driver、device
type、deterministic flags 或 cuDNN flags 不同，默认拒绝；
调用者可用 `--allow_env_drift` 显式接受。
接受后保存逐字段差异并将 run 标记为
`reproducibility_status=environment_drift`。该开关不能绕过训练语义或
major.minor 版本不匹配。

`num_workers`、输出路径和日志详细度是可自由改变的 launch 字段，每次调用
仍记录新旧值，但不需要 `--allow_env_drift`。这项豁免建立在 §12.2 的样本
顺序/增强去 worker 化和 §17 的跨 worker 等价测试上；测试失效时应修复数据
管线，而不是把 worker 数重新塞回 training contract。

`data_path` 只允许作为位置迁移字段改变；新路径下所有 manifest target/image
校验必须重新通过，并记录迁移 event。run 内 dataset bundle 路径不可替换。

同环境的精确 resume 测试覆盖 CPU、`num_workers=0` 且启用增强的最小路径；
另用 workers `0/2/4` 验证同一 epoch 的 sample id、batch 顺序和增强后 tensor
逐字节相等。协议不宣称不同硬件、依赖版本或 GPU kernel 的训练 tensor
bitwise 等价。

### 12.4 幂等 finalization

最后一个 epoch commit 不等于 run 完成。出现终态且没有预算延长时，无论来自
正常控制流还是 resume，都执行同一个幂等 finalization：

1. 按 12.1 修复 last、best、metrics 和 run sidecar，并验证 best checkpoint
   role、文件 hash 和 `best_metric_value`。
2. 若 `test.json` 已存在且其 best checkpoint hash、manifest/vocabulary hash、
   input/decoder contract 与当前终态完全相等，则复用；否则从 `best.pt` 执行
   test，并原子写入带上述身份和全部 test 指标的 `test.json`。
3. 从 committed artifacts 构造 `summary.json`，写入 stop reason、best/last/test
   hashes、最终指标和 verification/reproducibility status，再原子 replace。
4. 最后原子设置 `run.json.status=completed`；这是 run 的 final commit point。

任何步骤崩溃都保留 `last.pt` 权威状态；下次 resume 从步骤 1 重试，不进入
下一 epoch。若已 completed 的 `max_epochs` run 后来合法增加预算，先记录旧
summary hash 和 resume event，再把 status 改回 `running`；新终态会生成新的
test/summary，历史记录不得丢失。early-stopping run 不能重新打开训练。

## 13. 运行目录和 checkpoint

输出目录结构：

```text
<output_root>/
  <experiment_name>/
    <timestamp>-<contract_hash_prefix>-<run_id>/
      run.json
      dataset_bundle.json
      split_manifest.json
      vocabulary.json
      image_verification_index.json
      train_exclusions.json              # 仅 exclude_listed 成功 run
      train_exclusions.candidate.json    # 仅 fail policy 检出不可行样本
      metrics.jsonl
      checkpoints/
        last.pt
        best.pt
      test.json
      summary.json
```

timestamp 固定为 UTC `YYYYMMDDTHHMMSSZ`，`contract_hash_prefix` 是
training contract SHA-256 的前 12 个 hex。`run_id` 使用 `uuid4().hex` 的前
12 个 hex 字符，避免在 Windows 路径下浪费长度。这三者已足够消歧，最终仍以
`mkdir(exist_ok=False)` 防覆盖；碰撞即失败，由调用者重新启动。所有文件先
落到同目录临时文件，再用原子 replace 发布，只有 `metrics.jsonl` 采用可修复
append：每条 canonical JSON 加单个 `\n`，以 binary append 完整写入后
`flush + fsync`。它不宣称单次 append 原子；§12.1 用权威 `last.pt` 截断尾部
半行、补缺和拒绝冲突。

`dataset_bundle.json`、`split_manifest.json`、`vocabulary.json` 和
`image_verification_index.json` 是输入 bundle 的一次 run 级副本。preflight
结果合并进 `run.json`，不额外维护内容重复的
`preflight.json`。resume event 追加到 `run.json.resume_history` 数组，更新时
仍使用临时文件加原子 replace，历史项不得删除或改写。

成功的 `exclude_listed` run 还保存经过 canonical 校验的
`train_exclusions.json`；source path 只留在 launch audit 中。默认 `fail`
发现不可行样本时写出的 `train_exclusions.candidate.json` 不是自动授权排除，
只是让操作者能审阅并在下一次调用中显式选择。两者都不回写 dataset bundle
或 manifest。

新运行记录 source `dataset_bundle_path` 仅供审计；复制成功后，checkpoint 和
resume 一律解析 run 内相对路径，不回读原始 bundle。`data_path` 可以在 resume
时显式迁移，但所有内容校验必须重新通过。

checkpoint 至少包含：

checkpoint 不重复保存固定、冻结的 base model。`state_dict_scope` 固定为
`trainable_only`：linear probe 保存完整 task head；LoRA 保存 adapter 加完整
task head。expected key 集合由 method、LoRA config 和 task-head contract
生成，保存和加载都要求精确相等。恢复时先加载 approved base revision，再构建
adapter/head，最后应用 trainable state。这样 `last.pt` 每 epoch 原子更新不会
反复写数百 MB 不变 backbone；代价是 resume/inference 必须能从本地 cache 或
Hub 取得该固定 base revision。

```text
schema_version
protocol_version
run_id
checkpoint_role
state_dict_scope
trainable_state_dict
trainable_parameter_names
optimizer_state_dict
optimizer_parameter_names
epoch
global_step
early_stopping_state
best_metric_name
best_metric_value
best_epoch
best_updated
stop_reason
epoch_record
training_contract
training_contract_sha256
initial_launch_config
initial_launch_config_sha256
dataset_bundle_relpath
dataset_bundle_sha256
split_manifest_relpath
split_manifest_sha256
vocabulary
vocabulary_sha256
base_model_id
base_model_revision
base_model_weights_filename
base_model_weights_sha256
base_model_registry_evidence
backbone_config
input_contract
augmentation_contract_sha256
train_exclusions_relpath
train_exclusions_sha256
train_exclusions_count
base_seed
init_seed
seed_schedule_version
next_epoch
package_versions
```

`best.pt` 和 `last.pt` 都使用相同 schema。checkpoint loader 必须先验证 metadata，
再把 tensor state 应用到已构建模型。checkpoint 不重复嵌入完整 manifest；
resume 只接受 role 为 `last` 的文件，并通过 dataset bundle/manifest 相对路径
找到 run 副本、校验全部交叉 hash。完整 vocabulary 仍嵌入 checkpoint，因此
移动单个 checkpoint 后可以在不扫描原始 target 的情况下把“已经产生的 id”
映射为 token；它不包含冻结 base 权重，不能单独产生预测。推理至少还需要
checkpoint 固定并校验过的 base revision/weight SHA、输入契约和 trainable
state；继续训练则还需要完整 run artifact bundle 和同内容 `data_path`。

`summary.json` 保存 `best.pt` 和 `last.pt` 的文件 SHA-256，使 checkpoint 离开
原路径后仍可核对内容身份。

## 14. 指标和记录

本期不强制 W&B。结构化本地记录是协议要求，终端输出只是人类可读副本。

每个 epoch 至少记录：

- epoch 和 global step
- 实际处理样本数
- train loss
- 当前 learning rate
- validation 是否执行
- `val_CER_all`、`val_CER_feasible` 和 `val_CTC_loss_feasible`
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
- Python、PyTorch、Transformers、PEFT、TorchVision、NumPy、Albumentations
  和 OpenCV 版本
- 操作系统和设备
- protocol、training contract、每次 launch config、dataset bundle、manifest、
  vocabulary 和 augmentation hashes
- train/val/test 样本数
- 各 split 的 target 长度和 required frames 分布摘要
- train 排除文件 hash、数量、比例和长度分布
- approved base weight 文件名、size、内容 SHA-256 和 registry evidence
- image verification index 命中数、重算数、验证模式和 data verification status
- input geometry、完整 input contract 和官方预处理依据
- 可训练参数数量和名称前缀摘要
- resume history、环境漂移差异和 reproducibility status

## 15. 失败处理

以下情况必须在训练前失败：

- 配置类型或范围无效。
- `start_eval > max_epochs`。
- manifest schema、路径、split 或 group 隔离无效。
- 词表无效或出现 OOV。
- train 不可行集合不满足所选 policy 和显式排除文件。
- backbone revision 未固定或显式解析失败。
- augmentation profile 展开结果与契约不一致。
- patch grid 与 backbone config 不兼容。
- 输出目录不可创建。
- resume metadata 不匹配。

以下情况必须在训练过程中立即失败：

- loss 或 metric 非有限。
- batch 的 target length 与拼接 target 不一致。
- 模型输出时间步与输入契约不一致。
- checkpoint 原子写入失败。

`zero_infinity=False` 会暴露而不是掩盖异常；梯度裁剪也不能把非有限 loss
变回有限值。发生非有限 loss 时，中止当前 epoch，保留上一个完整 epoch 的
`last.pt` 和诊断信息。该 checkpoint 只是恢复点，不是自动修复：操作者必须先
定位数据、数值或环境原因，原 training contract 不允许通过 resume 偷换参数。

run 目录创建后发生的任何失败都必须把 `run.json.status` 原子更新为 `failed`，
记录阶段、异常类型和结构化上下文；创建前失败只向调用者返回错误。失败记录
不得删除既有 committed checkpoint 或伪造 summary 为成功。

错误信息必须给出具体字段、样本 id 或 checkpoint 字段，不能只返回底层
reshape、`KeyError` 或 `FileNotFoundError`。

## 16. README 和公共行为

README 必须与 v2 行为一致：

- 只描述实际执行的 train、validation 和 test 指标。
- 删除 `engine_test` 会打印样本对的错误陈述，除非实现确实提供该行为。
- 删除无 `__main__` 支持的 `python augments.py` 示例。
- 只展示规范的 patch grid CLI 语法，并说明过渡别名。
- 从平铺目录开始展示完整 `prepare-data` 和训练命令，不要求用户手写生成器。
- 说明显式 group regex、model revision 解析和 image hash 验证模式。
- 说明全量 CER 与可行子集 CER 的区别。
- 说明只有 LoRA/exact-grid 在 v2 可改变 `patch_cols`；linear probe 的时间轴
  固定为 backbone 原生列数。
- 展示默认 `fail` 的完整恢复路径：阅读 failed run 的 preflight 和
  `train_exclusions.candidate.json`；然后选择 LoRA 并增大 `patch_cols`，
  或审阅 candidate 后显式使用 `exclude_listed`。linear probe 不能靠增大宽度
  恢复，只能排除或改用 LoRA。
- 说明不同 LoRA 宽度导致的训练排除差异及比较限制。
- 说明 v2 使用 closed-corpus 词表及其 manifest 身份。
- 说明 linear probe/native-pad 与 LoRA/exact-grid 的首版风险边界。
- 说明 legacy checkpoint 不能直接 resume 到 v2。
- 明确 v2 建立可信基线，不宣称降低 CER 或提高吞吐。

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
- `linear_probe`/`lora` 分别派生 `native_pad`/`exact_grid`；任何显式
  `input_geometry` 输入都失败。
- training contract 的 Adam、增强和排除文件 hash/count 稳定。
- 只修改 worker、output/data location 或同内容排除文件的 source path，不改变
  training contract hash，但改变 launch config hash。

### 17.2 Manifest 和词表测试

- manifest hash 不受 JSON 键顺序和 sample 输入顺序影响。
- 重复 sample、跨 split group、路径逃逸、内容 hash 不匹配、缺失文件和空
  split 失败。
- bundle/manifest/vocabulary 的 dataset id 或交叉 hash 不一致时失败。
- verification index 缺项、多项或 path/size/hash 与 manifest 不一致时失败。
- `prepare-data` 对同一 fixture 和参数生成逐字节相同的三个身份 JSON：
  bundle、manifest 和 vocabulary。
- group regex 缺少命名捕获、不匹配文件或只有两个 group 时失败，且不发布
  最终 bundle 目录。
- manifest 不因 `patch_cols` 改变。
- cached/always image 验证模式产生正确的命中和重算统计。
- `always` 是默认；`cached` run 被标记为降级且不能作为可信基线验收。
- 同一 run resume 的 `always` 仍重算 image hash；替换为同 size/mtime 的不同
  内容会失败，不能沿用上次的 `content_verified`。
- vocabulary hash 稳定。
- 重复 token、blank 冲突、manifest hash 不匹配和 OOV 失败。
- `num_classes == len(tokens) + 1`，id 0 只属于 blank。

### 17.3 CTC 测试

- `[1, 2]` 需要 2 帧。
- `[1, 1]` 需要 3 帧。
- `[1, 1, 2, 2]` 需要 6 帧。
- `fail` 对 train 不可行样本在模型加载前失败。
- `fail` run 产生符合 schema、可直接作为下一次输入的 candidate 排除文件。
- `exclude_listed` 要求排除文件与 manifest/`patch_cols` 绑定，且 id 集合与
  实际不可行集合完全相等；path 变化但 canonical 内容相同不改变 contract。
- all-infeasible train fixture 即使全部显式列出也因保留样本为 0 而失败。
- val/test 不可行样本保留并进入分层统计。
- 构造 `minimum_frames(y) == patch_cols` 的边界样本，断言
  `zero_infinity=False` 时 loss 有限，且 task head 至少一个参数梯度非零。
- validation loss 只聚合可行样本，且不改变全量 CER 的分母。

### 17.4 数据管线测试

- 变长 target collate 生成正确的拼接 tensor 和 lengths。
- 最后一个小 batch 生成正确 input lengths。
- 评估按 target lengths 切片，不依赖 blank 哨兵。
- 超长 target 不导致其他样本被全局 padding。
- `sha256_epoch_order_v1` 对相同 seed/epoch/sample ids 产生稳定全排列，不受
  manifest 输入顺序影响。
- 启用 `staff_omr_train_v1` 后，workers `0/2/4` 产生逐字节相同的 sample id
  顺序、batch 边界和增强 tensor。
- 改变 epoch 或 sample id 会改变对应 sample seed；worker seed 不传给
  Albumentations。

### 17.5 模型契约测试

- linear probe 和 LoRA 对相同 patch grid 产生相同时间维。
- linear probe 使用 `native_pad` 且不插值位置编码；LoRA 使用 `exact_grid`
  且插值位置编码。
- linear probe 的 `patch_cols != native_cols` 失败；LoRA 可在兼容的多个
  `patch_cols` 上通过。
- exact-grid 的 position embedding 结果与
  `bicubic, align_corners=false, antialias=false` reference 一致；其他隐式
  行为在 preflight 失败。
- linear probe 只冻结 backbone。
- LoRA 只有 adapter 和 task head 可训练。
- 两种几何的非匹配 token count 均产生描述性错误。
- 不同 backbone image size 不依赖 `64` 或 `1024`。
- 改变增强参数或顺序会改变 training contract hash。
- 本地 tiny `ViTMAEForPreTraining(ViTMAEConfig(...))` checkpoint 能按 approved
  loading-info allowlist 被显式 `ViTModel` encoder 加载，产生一个 CLS token
  加完整 spatial tokens；loader 固定移除一个 prefix，拒绝 AutoModel 路径。
- alias approved revision 的 README/config blob oid、权重 size/LFS SHA-256、
  LFS pointer oid 或 processor absence 不匹配时失败，不对自然语言做 runtime
  判断。
- task-head 参数名、shape、bias、执行顺序和 state-dict schema 与
  `staff_omr_task_head_v1` 完全一致。
- 同环境、同 contract 的两个新模型在首 batch 前具有完全相同的 trainable
  state dict。

### 17.6 Checkpoint 测试

- checkpoint 包含完整 schema。
- checkpoint 的 trainable key 集合与 method contract 完全相等，不含冻结
  backbone；optimizer param group 也不含冻结参数。
- 打乱两个同 shape 参数的 checkpoint optimizer 名称顺序时，必须在
  `optimizer.load_state_dict()` 前拒绝。
- 不读取原数据目录也能把既有 id 映射到 token；缺少 approved base 权重时
  明确不能执行模型推理。
- checkpoint 不嵌入完整 manifest，run 副本 hash 不匹配时拒绝 resume。
- 删除原始 dataset bundle、保留 run 副本和同内容 `data_path` 后仍能 resume。
- `best` 和 `last` 不互相覆盖。
- 同配置两次运行生成不同 run 目录。
- `last.pt` 包含 validation 后更新的 best/patience/stop state 和 epoch record。
- 模拟 `last.pt` commit 后、best/metrics 更新前崩溃，resume 能从 last 修复
  sidecar 和滞后的 best，再进入下一 epoch。
- 在 `metrics.jsonl` 尾部注入半行时，resume 截断并补 last record；更早 epoch
  gap、重复 epoch 内容冲突或 metrics 领先于 last 时拒绝。
- 模拟终态 last commit 后、test/summary 前崩溃，resume 不训练新 epoch，
  幂等补齐 test、summary 并把 status 置为 completed。
- CPU、`num_workers=0`、增强开启时，epoch 边界 resume 与不中断运行的下一
  epoch tensor 完全一致。
- resume 把 `num_workers` 从 0 改为 2 无需 `--allow_env_drift`，下一 epoch
  的样本顺序和增强 batch 仍逐字节一致，并记录 launch 变化。
- protocol、vocabulary、manifest、排除文件、revision、base weight SHA 或
  patch grid 不匹配时拒绝 resume。
- `checkpoint_role=best` 明确拒绝 resume；只有同一 run 的 last 可恢复。
- `max_epochs` 只能增大；package patch/device 漂移必须显式
  `--allow_env_drift` 并留下审计记录；worker 数只记录，major.minor 漂移
  仍失败。
- v2 loader 无条件拒绝 legacy state dict。
- summary 中的 checkpoint 文件 hash 与实际文件一致。

### 17.7 CPU 集成测试

使用少量临时图片和 dummy backbone：

1. 运行 `prepare-data` 生成完整 dataset bundle。
2. 完成一次预检。
3. 训练至少一个 epoch。
4. 写入 `last` 和 `best`。
5. 从 `best` 执行 test。
6. 从 `last` 增大 `max_epochs` 并继续一个 epoch。
7. 验证 run metadata、metrics、resume history 和 summary 完整。

## 18. 验收标准

实现只有在以下条件全部满足时才能声明 v2 完成：

1. train 中不存在被静默忽略的 CTC 不可行样本；显式排除可审计。
2. 相同 manifest、词表和 training contract 产生相同的三个 identity hash；
   修改增强、method/派生输入几何或排除文件内容必然改变 training contract
   hash，而只修改 `num_workers` 不改变任何训练输入。
3. train/val/test 的 group 集合两两不相交。
4. checkpoint 在不扫描原始 target 文件的情况下能够把既有预测 id 映射为
   token，并明确记录产生预测所依赖的 base weight SHA-256。
5. `start_eval > max_epochs` 在加载模型前失败。
6. 相同配置的两次运行不覆盖任何文件。
7. method 唯一派生 linear probe/native-pad 或 LoRA/exact-grid，二者都由
   显式 input contract 驱动；runtime geometry 逻辑不含 `64` 或 `1024`
   字面量，位置编码插值经过 reference 校验。
8. 完整 run artifact bundle 配合同内容数据和 approved base
   revision/weight SHA，能在 epoch 边界从 last 恢复训练，并允许只增大
   `max_epochs`。
9. v2 loader 明确拒绝 legacy checkpoint。
10. 全量 CER、可行子集 CER、可行 validation loss 和容量统计同时落盘。
11. README、CLI help 和实际参数语义一致。
12. 从平铺 fixture 运行 `prepare-data` 后，无手工生成文件即可进入训练预检。
13. 用于声明可信基线的 run 对全部 image 执行本次内容 hash 校验。
14. task-head schema、optimizer 名称映射、epoch commit 和 finalization state
    machine 有机器可执行的契约测试。
15. 所有新增测试在 CPU、无网络环境通过。
16. 相同 seed/epoch 下 workers `0/2/4` 的样本顺序和增强后 batch 逐字节相同。

## 19. 后续阶段

### 19.1 v2.1 协议加固

以下能力不阻塞 v2 可信基线：

- batch 中间恢复。
- 序列化第三方库内部 RNG 状态。
- GPU、跨设备和跨依赖版本的完整训练 bitwise resume 保证。
- image verification index 的签名、共享索引或远程数据存储适配。
- legacy checkpoint 转换。

这些能力只有在真实运维需求证明其成本合理后才进入 v2.1；v2 的确定性 epoch
seed schedule 和完整 epoch checkpoint 已覆盖常见的“增加训练轮数”恢复场景。

### 19.2 输入几何证据

首次真实 linear probe 基线完成后，使用相同 manifest、词表、训练排除、
初始化和优化协议，一次性比较 `native_pad` 与 `exact_grid`。报告全量 CER、
可行 CER、训练 loss 和 input contract。只有证据显示插值没有造成不可接受
回归，后续协议版本才可开放 `linear_probe + exact_grid` 用户配置；v2 公开
入口继续拒绝 `input_geometry` 覆盖，不能静默换默认。

### 19.3 性能阶段

可信基线完成后，单独设计性能协议。候选项包括：

- AMP 与 GradScaler。
- `pin_memory` 和 non-blocking transfer。
- Windows/Linux 分别选择 worker、persistent workers 和 prefetch。
- input lengths 分配与 host/device 传输优化。
- 避免每个 batch 显式创建 LSTM 零状态。

每项必须报告吞吐、峰值显存、训练 loss 数值稳定性和固定验证集 CER parity。
没有测量时不承诺加速倍数。

### 19.4 优化器阶段

AdamW、WSD、warmup、cosine 和梯度裁剪使用独立协议版本。比较时固定 split、
词表、初始化、样本数、batch size 和随机种子。不得把 `full_page_omr`
参数直接复制后视为 staff-level 默认值。

### 19.5 模型消融阶段

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
