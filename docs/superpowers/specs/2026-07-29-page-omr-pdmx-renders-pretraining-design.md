# page-omr-pdmx-renders 原生流式预训练规格

状态：2026-07-29 方案 A 已获用户确认，等待用户审阅本规格后实施。

数据源：
[tobiashornbogen/page-omr-pdmx-renders](https://huggingface.co/datasets/tobiashornbogen/page-omr-pdmx-renders)，
固定 revision `7da3ae5237963e57a8fe1c6ee375b1f10af34a09`。

## 1. Linus 判断

### 这是真问题吗

是。当前 full-page OMR 的真实页数据规模小，排版、字体、扫描退化和符号覆盖有限。
PDMX renders 提供约 458K 个合成页面、两个 renderer 和大量排版变化，能解决预训练阶段的覆盖问题。

### 有没有更简单的办法

有，但只适合试验：把少量 tar shard 离线转换成当前 Arrow schema。
该方案会复制接近 90 GB 的数据，破坏流式读取优势，也无法形成可长期维护的全量训练路径，因此不作为生产设计。

生产方案是新增一个独立的 WebDataset 预训练 DataModule。现有 Polish Scores、Mozarteum 和
FP GrandStaff 的 Arrow loader 保持原样，不把 WebDataset 特例塞进 `_ArrowOMRSource`。

### 会破坏什么

直接把 Hugging Face dataset id 写进现有 `finetuning.json` 会破坏以下契约：

1. 当前 loader 需要可 `len()`、可整数索引的 Arrow dataset；PDMX 是 tar 流。
2. 当前字段是 `image`、`transcription`；PDMX 字段是 `.image.png`、`.kern.txt`。
3. 当前训练假定 `train/val/test` 三个 split；PDMX 只有 train 和 validation。
4. 当前词表生成会遍历数据并使用无序 `set`；对约 90 GB 数据既昂贵又不可复现。
5. 当前 checkpoint 的 embedding 和输出层大小与词表绑定；扩大词表后不能直接严格恢复旧 checkpoint。
6. 当前 curriculum 会把配置数据当作少量真实数据，并混入在线 Verovio 样本；PDMX 本身就是大规模合成预训练集，语义不成立。

因此本规格明确建立新的数据、词表、checkpoint 和评测边界，不伪装成第四个 CL 微调配置。

## 2. 已确认事实

### 官方数据

- license 为 CC BY 4.0，底层 PDMX 乐谱为 public domain。
- `verovio` config 有 train 和 validation：
  - train 位于 `shards_pdmx_a1_broadened_518x728/*/pdmx-*.tar`；
  - validation 为 `curated_validation_pdmx.tar`。
- `mscore` config 只有 train，位于
  `shards_pdmx_a1_mscore_518x728/*/pdmx-*.tar`。
- train 图像标称为 `518 x 728`；固定 revision 的 curated validation 共 21 个样本，
  图像尺寸可变，不能假定也是 `518 x 728`。
- tar 中必需成员为 `<key>.image.png`、`<key>.kern.txt` 和
  `<key>.source.txt`；`<key>.fill.txt` 为可选密度 metadata。
- 数据按 voice count 和 page density 分桶。两个 renderer 的同一乐谱不是泄漏，
  而是有意保留的视觉变体。
- 官方声明已按 composer/title 过滤 SMB、OpenScore Quartets/Lieder 和
  Polish Digital Scores，但本项目仍需独立检查 PDMX train 与 curated validation 的
  `.source.txt` 是否相交。
- 官方没有发布 test split。PDMX validation 只能选择预训练 checkpoint，不能代表真实世界最终性能。

### 当前项目

- `parse_kern_file(..., tokenization_mode="bekern")` 是现有唯一 BeKern 转换语义。
- 模型当前固定 `maxlen=7512`、batch size 1、目标输入 `1024 x 1024`。
- `convert_img_to_tensor()` 会做一次 RGB 转换和 `1024 x 1024` resize。
- PDMX Verovio 图像已经包含 Augraphy degradation；再次应用当前随机透视、弹性形变和模糊会产生双重退化。
- 当前三个词表大小为：
  - Polish Scores：215；
  - Mozarteum：191；
  - FP GrandStaff：181。
- Polish Scores 仍是当前最全面的单一词表，但不是三个现有词表的全集：
  - Mozarteum 额外包含 `*staff1`、`*staff2`、`=:|!;`、`==;`；
  - FP GrandStaff 额外包含 `*M6/16`、`88`。
- 当前 `make_vocabulary()` 使用 Python `set` 的遍历顺序分配 id，同一符号集合不能保证生成相同 id。

## 3. 目标

1. 原生读取固定 revision 的 PDMX WebDataset，不转换为 Arrow，不在训练时访问不固定的 `main`。
2. 同时支持 Verovio 和 MuseScore train shards，并显式控制 renderer 比例。
3. 保持现有三个 Arrow loader 和 legacy 配置行为不变。
4. 以 Polish Scores 215-token 映射作为不可变 id 前缀，再做确定性的并集扩展。
5. 训练开始前完成全部词表发现；训练过程中禁止动态扩容和静默未知 token。
6. 让 legacy checkpoint 能以 token 名称迁移到扩展词表，但禁止词表不一致的完整恢复。
7. 为有限或循环的流式数据建立明确的 virtual epoch、shuffle、worker 和恢复语义。
8. 使用官方 curated validation 选择预训练 checkpoint，不伪造 test。
9. PDMX 预训练结束后，以同一扩展词表在 Polish Scores/Mozarteum 上做真实域微调。
10. 记录足以复现数据身份、词表身份、样本顺序和训练拓扑的 manifest。

## 4. 非目标

- 本规格不把 PDMX、Polish Scores 和 Mozarteum 混在同一个 batch stream。
- 本规格不新增 Polish Scores 与 Mozarteum 的多源 Arrow loader；该需求需要独立规格。
- 本规格不修改已有 legacy `.npy` 词表或已有 checkpoint。
- 本规格不声称 PDMX validation 指标等于真实扫描乐谱指标。
- 本规格不实现 DDP、多节点训练或 batch size 大于 1。
- 本规格不改变 SMT encoder、adaptor、decoder 层数、`d_model`、attention backend 或 `maxlen`。
- 本规格不同时实验 runtime augmentation、分辨率、optimizer 或 encoder freeze schedule。
- 本规格不根据 validation/test 标签动态补充词表。
- 本规格不允许训练过程隐式下载 90 GB 数据。

## 5. 总体架构

新增 `PDMXPretrainingDataModule`，并在训练入口注册独立 regime `PDMX`：

```text
fixed Hugging Face revision
    -> offline prepare + dataset manifest
    -> deterministic vocabulary scan
    -> frozen FullPageOMR_BeKern_v1 manifest
    -> local WebDataset streams
         -> Verovio stream --\
                              -> deterministic 50/50 mix
         -> MuseScore stream -/
    -> shared BeKern parser
    -> existing image tensor contract
    -> SMT pretraining
    -> curated PDMX validation
    -> weights-only downstream real-data fine-tuning
```

所有 PDMX 特有 tar、renderer mixing、stream resume 和 source metadata 逻辑放在独立模块，
不向 `_ArrowOMRSource`、`RealDataset` 或 `CurriculumTrainingDataset` 增加格式分支。

`ExperimentConfig.data` 使用 tagged union：

- 旧配置缺少 `type` 时按 `arrow` 解析，保持兼容；
- 新配置必须显式使用 `type: "pdmx_webdataset"`；
- Arrow 配置不能接受 PDMX-only 字段，PDMX 配置也不能接受 `reduce_ratio` 或 `skip_steps`。

DataModule 暴露统一能力：

```text
has_validation_split = true
has_test_split = false
encoder_unfreeze_step = 0
curriculum_step_offset = 0
```

训练入口根据 `has_test_split` 决定是否执行 test。不得让 PDMX 的 `test_dataloader()`
返回 validation，也不得通过捕获 “missing test” 异常跳过测试。

## 6. 数据准备与身份

### 6.1 下载和本地解析

数据准备与训练分成两个命令：

1. prepare 命令从固定 revision 下载被选中的 shards 到标准 Hugging Face cache；
2. training 只使用 `local_files_only=True` 解析已存在 snapshot。

训练启动时缺少任一文件必须在模型构造前失败，并列出缺失 logical path。
训练进程不回退到网络、不切换 revision，也不自动选择较少的 shard。

物理 cache 路径不进入可移植 manifest。manifest 只记录 dataset id、revision、repo 内 logical path、
文件大小和 SHA-256；运行时再由 Hugging Face cache 解析本机路径。

### 6.2 Dataset manifest

prepare 阶段生成版本化 JSON manifest，至少包含：

```text
schema_version
dataset_id
dataset_revision
license
selected_renderers
renderer_weights
train_shards[
  renderer, logical_path, sha256, bytes, sample_count,
  voice_bucket, density_bucket
]
validation_shard[
  logical_path, sha256, bytes, sample_count
]
tokenization_mode
max_sequence_length
sequence_length_summary
excluded_samples[
  shard, key, reason
]
train_validation_source_overlap
scan_tool_version
manifest_sha256
```

`manifest_sha256` 对移除自身字段后的 canonical JSON 计算。所有 key 排序，UTF-8 编码，
数组顺序按 logical path 固定。

prepare 必须扫描 tar 成员并验证：

- 每个 key 恰好有一个 `image.png`、`kern.txt` 和 `source.txt`；
- `fill.txt` 可以缺失；
- 同一 renderer 内 key 不重复；
- PNG 可解码，Kern/source 使用严格 UTF-8；
- tokenized target 非空；
- `[<bos>, ..., <eos>]` 总长度不超过 7512；
- train 与 curated validation 的 normalized `source.txt` 集合不相交。

损坏、缺配对、重复 key、source overlap 默认都是 fatal。超过 7512 的页面可以被显式排除，
但 key、原因和数量必须写入 manifest；训练时的排除集合必须与 manifest 完全一致。
不能运行时遇到长样本才静默截断。

### 6.3 Source mixing

v1 生产配置同时启用 `verovio` 和 `mscore`，按样本严格使用 `0.5/0.5` 权重。
权重必须是有限正数且总和为 1，不做隐式归一化。

renderer 内部使用 manifest 中的全部 bucket，并保持该 renderer 的自然 shard/sample 分布。
v1 不对 `voices_9p`、`triplet_heavy` 或其他 bucket 单独过采样；没有误差证据前不增加特殊策略。

两个 renderer 指向同一 `.source.txt` 的页面允许同时进入 train，因为它们是视觉渲染变体。
同一 source 不能出现在 validation。

每个 renderer 都按有限、无放回的 shard cycle 遍历。一个 cycle 内每个合格样本最多出现一次；
某个 renderer 被消费完但 `max_steps` 尚未到达时，递增该 renderer 的 `source_cycle`，用
`(seed, renderer, source_cycle)` 派生新顺序后继续。不能用无限有放回抽样掩盖实际数据
覆盖率。`source_cycle` 必须进入 checkpoint cursor 和 run protocol。

## 7. 配置契约

新增设计目标配置：

```json
{
  "data": {
    "type": "pdmx_webdataset",
    "dataset_id": "tobiashornbogen/page-omr-pdmx-renders",
    "dataset_revision": "7da3ae5237963e57a8fe1c6ee375b1f10af34a09",
    "dataset_manifest": "experiments/full_page_omr/config/Page_OMR_PDMX/dataset-manifest.v1.json",
    "vocab_manifest": "experiments/full_page_omr/vocab/FullPageOMR_BeKern_v1.json",
    "renderer_weights": {
      "verovio": 0.5,
      "mscore": 0.5
    },
    "batch_size": 1,
    "num_workers": 8,
    "tokenization_mode": "bekern",
    "steps_per_epoch": 10000,
    "shuffle_buffer": 2048,
    "seed": 3407,
    "runtime_augmentation": false
  }
}
```

上述 `num_workers` 是参考值，不宣称为性能最优值；实际生产值必须进入 run protocol，
完整恢复要求保持一致。

验证规则：

- dataset id、40 位 revision、manifest path、vocab manifest path 必须非空；
- manifest 内 id/revision 必须与配置完全一致；
- batch size 必须等于 1；
- `num_workers >= 0`；
- `steps_per_epoch > 0`；
- `shuffle_buffer >= 1`；
- seed 必须是非负整数；
- tokenization mode 必须为 `bekern`；
- v1 的 runtime augmentation 必须为 false；
- renderer key 必须恰好对应 manifest 已选择的 renderer。

## 8. Tokenization 和图像处理

### 8.1 单一 tokenization 入口

`.kern.txt` 按严格 UTF-8 解码后，只调用现有
`parse_kern_file(kern, tokenization_mode="bekern")`，再添加 `<bos>` 和 `<eos>`。

manifest scanner、vocabulary builder、train dataset 和 validation dataset 必须调用同一个函数。
禁止为扫描速度复制一个“近似 tokenizer”，否则词表发现和训练会发生分叉。

任何 tokenizer 规则变化都需要：

1. 新 tokenization protocol version；
2. 重新扫描完整 manifest；
3. 新 vocabulary version 和 digest；
4. 禁止旧 run 完整恢复。

### 8.2 图像

- `image.png` 解码后转换为 RGB。
- v1 直接调用现有 `convert_img_to_tensor()`，只做一次 `1024 x 1024` resize。
- PDMX 不使用 `reduce_ratio`。
- train 和 validation 使用相同的确定性 resize；validation 的可变原始尺寸必须被支持。
- v1 不调用 `augment()`，避免对已包含 Augraphy degradation 的 Verovio 页面双重破坏。
- 首批 run audit 保存每个 renderer 至少一个样本的 key、原始尺寸、最终 tensor shape，
  以及可人工查看的处理后图像。

后续若要比较 runtime augmentation，必须建立新 protocol，一次只改变该变量。

## 9. 词表协议

### 9.1 不可变 Polish 前缀

新词表名称为 `FullPageOMR_BeKern_v1`。它不是重新排序后的 Polish 词表，而是保留当前
Polish Scores 215 个 token 的精确 id：

```text
id 0..214 = 当前 Polish_Scores_BeKern 的原始映射
```

当前特殊 token id 必须保持：

```text
<pad> = 0
<s>   = 44
<bos> = 100
<b>   = 132
<eos> = 183
<t>   = 29
```

按 ordered token array 的 canonical JSON 计算，当前 215-token 前缀 digest 为：

```text
3821e7f0d5defd55fe73ce8b7f229f48f688bc490bdae854bc824a6145dac49b
```

实现必须同时校验当前 `.npy` 文件内容和上述逻辑 digest。文件存在但映射不同也必须失败。

### 9.2 现有数据集兼容扩展

为保证 PDMX checkpoint 后续无需再次 resize decoder 就能在当前三个数据集上微调，
先把 Mozarteum/FP GrandStaff 相对 Polish 缺失的六个 token 按 UTF-8 byte order 追加：

```text
215 *M6/16
216 *staff1
217 *staff2
218 88
219 =:|!;
220 ==;
```

因此 PDMX 扫描前的项目级 seed vocabulary 固定为 221 个 token。

### 9.3 PDMX 扩展

只扫描 manifest 选中的 PDMX train labels：

```text
pdmx_new = tokens(PDMX train) - tokens(seed vocabulary)
```

`pdmx_new` 不做 Unicode normalization、不改大小写，按 `token.encode("utf-8")` 排序后从 id 221
开始连续追加。最终大小为 `221 + N`；在完整 train shard 扫描完成前不得猜测 N。

validation 只做覆盖检查，不参与扩展。validation 出现 OOV 时 prepare 失败，不能从 validation
标签补词，也不能在训练时映射为未知字符。

v1 不新增 `<unk>`。OMR 输出是封闭符号转写，漏掉 token 表示 vocabulary manifest 或 dataset
revision 错误，应立即失败并报告 shard、sample key 和全部 OOV token。

### 9.4 Artifact 和 digest

JSON manifest 是新词表的唯一 source of truth，至少记录：

```text
schema_version
name
tokenization_mode
base_name
base_size
base_digest
ordered_tokens
token_provenance
source_dataset_manifests
vocab_sha256
```

`w2i` 必须从 `ordered_tokens` 推导，`i2w` 必须是其严格逆映射。兼容旧代码需要的 `.npy`
文件只能由该 JSON 生成，不能反向覆盖 JSON。

`vocab_sha256` 使用：

```python
sha256(
    json.dumps(
        ordered_tokens,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
)
```

未来 `v2` 扩展必须以整个 v1 ordered array 为前缀，只能追加，不能插入、删除或重排。

### 9.5 禁止旧 vocabulary builder

PDMX 路径不得调用当前基于无序 `set` 的 `make_vocabulary()`。
现有函数可以继续服务 legacy 配置，但新 manifest builder 必须独立、确定且幂等：

- 同一输入重复运行得到 byte-for-byte 相同 JSON；
- shard 枚举顺序变化不改变 token id；
- worker 数变化不改变 token id；
- manifest revision 或 tokenizer version 变化会改变输入身份并强制重新扫描。

## 10. Checkpoint 兼容性

### 10.1 Checkpoint metadata

所有新 checkpoint 必须嵌入：

```text
vocab_name
vocab_size
vocab_sha256
vocab_schema_version
dataset_manifest_sha256
dataset_revision
tokenization_mode
```

模型构造、完整恢复、weights-only 迁移和评测都必须在加载大 tensor 前校验这些字段。

### 10.2 完整恢复

`from_checkpoint` 只允许以下值全部相同：

- vocabulary digest；
- dataset manifest digest；
- tokenization mode；
- model maxlen/resolution；
- renderer weights、shuffle buffer、seed；
- world size、num_workers、steps per epoch；
- optimizer protocol。

任一项不同都拒绝完整恢复。不能用 `strict=False` 隐藏 embedding/output head shape 不一致。

### 10.3 Weights-only token 迁移

新增显式 vocabulary-aware 迁移工具。它接收 source checkpoint、source vocabulary manifest 和
target vocabulary manifest，并按 token 字符串复制三组 token-dependent 参数：

```text
decoder.embedding.weight[token_id, :]
decoder.out_layer.weight[token_id, :, :]
decoder.out_layer.bias[token_id]
```

规则：

1. 先构造完整 target model，使新 token 行使用模型原生初始化。
2. 对 source/target 的每个同名 token，把 source 行复制到 target 行。
3. 所有非 token-dependent 参数必须名称和 shape 严格一致后复制。
4. target 中新增 token 保持新初始化。
5. source 中存在而 target 中缺失的 token 默认 fatal。
6. 输出 common/new/dropped token 和 tensor 数量的 JSON migration report。
7. optimizer、scheduler、global step 和 data cursor 全部丢弃，新 run 从 step 0 开始。

即使 Polish 215-token id 已被保留，迁移实现仍按 token 名称工作，不能依赖“前 215 行直接复制”
这一偶然捷径。这样同一工具也能正确迁移 id 顺序不同的 Mozarteum/FP GrandStaff checkpoint。

legacy checkpoint 尚无 JSON vocabulary manifest。迁移前先用一次性转换命令读取配对的
`w2i.npy/i2w.npy`，校验完整双射、连续 id 和两个文件的 SHA-256，再生成只读 legacy source
manifest。迁移工具不直接信任单个 `.npy`，也不从 checkpoint tensor shape 猜 token 顺序。

### 10.4 下游微调

PDMX checkpoint 微调 Polish Scores 或 Mozarteum 时继续使用完整
`FullPageOMR_BeKern_v1`，不得缩回 dataset-specific 215/191-token 词表。
目标数据未使用的输出类别保留；若模型错误地产生这些 token，正常计入识别错误。

legacy run 继续使用原词表和原 checkpoint，不在中途迁移。
下游新增独立 `pdmx_finetuning.json` 配置引用统一词表；已有 `finetuning.json` 不改名、不改
词表，以免 legacy checkpoint 在原命令下改变模型 shape。

## 11. 流式顺序、virtual epoch 与恢复

### 11.1 Virtual epoch

PDMX train 是流式混合，不把“遍历一次所有 tar”伪装成 Lightning epoch。

```text
1 virtual epoch = steps_per_epoch 个全局 training batches
```

batch size 固定为 1，因此参考配置每个 virtual epoch 消费 10,000 个样本。
DataLoader 必须实际停止在该边界，`len(train_dataloader)`、Lightning 进度和 protocol metadata
都报告同一个值，不能因 worker 数量而乘倍。

reference protocol 每个 virtual epoch 运行一次 21 页 curated validation。
训练总预算仍由显式 `max_steps` 决定，不由 dataset exhaustion 隐式结束。

### 11.2 确定性

固定 manifest、seed、virtual epoch 和 worker id 后：

- shard shuffle 使用确定性算法；
- sample shuffle buffer 使用独立确定性 RNG；
- renderer 选择使用独立确定性 RNG；
- teacher-forcing corruption 由
  `(seed, virtual_epoch, renderer, shard, sample key, occurrence index)` 派生，
  不依赖 worker 调度的全局 NumPy RNG；
- validation 保持 manifest 原顺序，不 shuffle；
- DataLoader 明确使用 ordered delivery。

renderer、shard、sample 和 teacher-forcing RNG 不能共享一个隐式状态，否则修改 buffer 大小会
连带改变 renderer 比例。

### 11.3 恢复

checkpoint 额外保存：

```text
virtual_epoch
consumed_in_virtual_epoch
samples_seen
stream_protocol_version
stream_topology
```

恢复时用相同 topology 重建该 virtual epoch 的确定性流，并在主进程丢弃已经消费的 batch，
从第一个未消费 sample 继续。worker 预取不增加 `samples_seen`。

为使上述行为可验证，v1 使用 `prefetch_factor=1`、ordered delivery，并在 virtual epoch
边界重建 worker。完整恢复要求 `num_workers` 和软件版本不变；不同 topology 只允许
weights-only 新 run。

测试必须证明对 `num_workers=0` 和 `num_workers=2`：

- 连续运行的 sample key 序列；
- 在任意非边界 step 保存并恢复后的 sample key 序列；
- 对应 teacher-forcing decoder input

三者在恢复点之后完全一致。无法达到这一点时，不得宣称 mid-epoch exact resume；
实现必须退回“只允许 virtual-epoch 边界完整恢复”，不能留下模糊保证。

v1 只承诺单进程、单 GPU。DDP 的 rank split 和全局恢复顺序另行设计。

## 12. 训练与评测协议

### 12.1 Reference pretraining

reference run 使用：

```text
regime = PDMX
protocol_version = full_page_omr_pdmx_pretrain_v1
dataset revision = 7da3ae5237963e57a8fe1c6ee375b1f10af34a09
vocabulary = FullPageOMR_BeKern_v1
batch size = 1
resolution = 1024
maxlen = 7512
encoder_unfreeze_step = 0
runtime augmentation = false
validation split = curated PDMX validation
checkpoint monitor = val_SER_v2
test = disabled because no test split exists
```

reference run 从 foundation encoder 和新初始化 decoder 开始。Polish 词表作为输出协议，
不代表从 Polish checkpoint 初始化权重。

训练步数、learning rate 和 WSD schedule 必须显式记录，但本数据接入规格不宣称某个预算最优。
生产 run 开始前另行批准训练预算；smoke test 预算不能冒充生产配方。

### 12.2 评测含义

PDMX validation 记录 `val_CER_v2`、`val_SER_v2`、`val_LER_v2`、EOS 命中率和截断数。
best checkpoint 只使用 `val_SER_v2`。

训练结束后不调用 `trainer.test()`。最终实用价值只能通过同一真实域微调协议比较：

```text
control:
foundation -> FullPageOMR_BeKern_v1 初始化 -> Polish/Mozarteum fine-tune

treatment:
foundation -> PDMX pretrain -> Polish/Mozarteum fine-tune
```

control 和 treatment 必须使用相同扩展词表、相同真实数据、相同微调预算和相同评测代码。
否则“是否受益于 PDMX”会与“是否扩大词表”混为一个变量。

test split 只在最终真实域模型上运行，不用于选择 PDMX checkpoint 或微调超参数。

## 13. 日志与审计

本地 run protocol 和 W&B 至少记录：

- dataset id、revision、manifest SHA-256；
- 每个 renderer 的 shard/sample 数与声明权重；
- 实际消费的 renderer、voice bucket、density bucket 计数；
- vocab name、size、SHA-256 和 base digest；
- tokenization mode；
- steps per epoch、shuffle buffer、seed；
- virtual epoch、samples seen、worker topology；
- resolution、maxlen、augmentation mode；
- excluded/invalid/overlength sample 数；
- validation sample 数和 source-overlap 审计结果；
- checkpoint 来源和 vocabulary migration report；
- `webdataset`、PyTorch、Lightning、datasets、huggingface_hub 版本。

实际 renderer 比例每个 virtual epoch 都应记录。有限样本下允许随机波动，但全 run 比例与
0.5 的偏差超过预先定义的统计门槛时必须报警，不能静默变成单 renderer 训练。

## 14. 错误处理

以下情况在模型训练前失败：

- dataset revision 未固定或与 manifest 不一致；
- 本地 shard 缺失、大小或 SHA-256 不一致；
- vocab manifest digest 不一致；
- validation OOV；
- train/validation source overlap；
- config 使用未知 renderer 或非法权重；
- 完整恢复的 stream/model/vocab protocol 不一致。

以下情况在读取具体样本时失败并报告 renderer、shard、key：

- tar pairing 损坏；
- PNG 解码失败；
- Kern UTF-8 或 tokenization 失败；
- runtime 发现 manifest 未声明的 OOV 或 overlength；
- metadata 中实际 sample count 超出 manifest。

禁止无限 `warn + skip`。若确实需要排除坏样本，应先重新运行 prepare，把排除项写入新 manifest
并产生新 digest。

## 15. 涉及文件

预计新增：

- `experiments/full_page_omr/pdmx_data.py`
- `experiments/full_page_omr/pdmx_manifest.py`
- `experiments/full_page_omr/utils/vocab_manifest.py`
- `experiments/full_page_omr/config/Page_OMR_PDMX/pretraining.json`
- `experiments/full_page_omr/config/Polish_Scores/pdmx_finetuning.json`
- `experiments/full_page_omr/config/Mozarteum/pdmx_finetuning.json`
- `experiments/full_page_omr/vocab/FullPageOMR_BeKern_v1.json`
- `tests/test_full_page_omr_pdmx_data.py`
- `tests/test_full_page_omr_vocab_manifest.py`
- `tests/test_full_page_omr_vocab_migration.py`

预计修改：

- `experiments/full_page_omr/config/ExperimentConfigWrapper.py`
- `experiments/full_page_omr/finetune.py`
- `experiments/full_page_omr/smt_trainer.py`
- `pyproject.toml`
- `uv.lock`
- `tests/test_full_page_omr_config.py`
- `tests/test_full_page_omr_checkpoint_export.py`

`webdataset` 作为锁定依赖加入项目。新逻辑应位于独立模块，不继续扩大已经承担 Arrow、
curriculum 和 online generator 逻辑的 `data.py`。

## 16. 测试

### 16.1 无网络单元测试

用本地生成的 tiny tar fixture 覆盖：

1. `image.png/kern.txt/source.txt` 正常配对；
2. `fill.txt` 可选；
3. 缺字段、重复 key、坏 PNG、坏 UTF-8 明确失败；
4. Verovio/MuseScore mixing 可复现；
5. validation 顺序固定；
6. variable-size validation 图像得到固定 tensor shape；
7. overlength 和 OOV 不被截断或跳过；
8. `steps_per_epoch` 不随 worker 数量变化；
9. mid-epoch resume 的 key 和 decoder input 等价；
10. PDMX 模式不会调用 test。

### 16.2 Vocabulary golden tests

1. Polish 215-token ordered digest 与本规格一致。
2. 所有 0..214 id 不变，特殊 token id 不变。
3. 六个现有兼容 token 固定为 215..220。
4. PDMX OOV 无论扫描顺序如何都按 UTF-8 byte order 追加。
5. 重复构建得到 byte-for-byte 相同 JSON。
6. validation OOV 不扩展词表而是失败。
7. `w2i` 与 `i2w` 严格互逆。

### 16.3 Checkpoint migration tests

1. Polish checkpoint 的所有同名 embedding/head 行精确迁移。
2. Mozarteum/FP checkpoint 通过 token 名称迁移，不依赖旧 id。
3. 新 token 行保持 target model 初始化。
4. 非 token 参数逐项严格复制。
5. optimizer state 不进入 weights-only 新 run。
6. vocabulary digest 不同的完整恢复在 tensor load 前失败。

### 16.4 可选官方数据集成测试

显式 opt-in 测试读取固定 revision 的 curated validation 和每个 renderer 至少一个 train shard：

- 校验官方字段和 sample count；
- 运行相同 tokenizer；
- 验证无 OOV、无 source overlap；
- 解码并前向一批；
- 完成一次 21 页 validation。

该测试缺少本地数据时标记 `not-run`，不能偷偷联网。

## 17. 验收标准

1. 现有 Polish、Mozarteum、FP GrandStaff 配置和 loader 测试行为不变。
2. PDMX 训练只读取固定 revision 和 manifest 中列出的本地 tar。
3. 完整 manifest 对所有选中 shards 记录 hash、sample count、token length 和 source-overlap 结果。
4. train/validation 所有 runtime token 都存在于冻结词表，OOV 数为 0。
5. Polish 215 个 id 完全保留，最终词表大小为 `221 + N`，N 由完整 train 扫描报告。
6. PDMX 新 token 的 id 与 tar/shard/worker 扫描顺序无关。
7. Verovio 和 MuseScore 都实际进入训练，长期样本比例符合 `0.5/0.5` 协议。
8. batch size 1、1024 resolution、7512 maxlen 契约保持。
9. 每个 virtual epoch 恰好产生 `steps_per_epoch` 个全局 batch。
10. 固定 topology 下 mid-epoch resume 通过 sample-key/decoder-input 等价测试；否则实现明确降级为只支持 epoch 边界恢复。
11. PDMX run 只运行 validation，不创建或复用假 test split。
12. checkpoint 带 dataset/vocabulary/stream digest；不兼容完整恢复在训练前失败。
13. legacy checkpoint 能生成完整 token-aware migration report，并以 weights-only 启动新 run。
14. 一次短 smoke run 能消费两个 renderer、保存 checkpoint、恢复并完成 21 页 validation。
15. PDMX 带来的收益最终以同词表、同微调协议的 control/treatment 对照证明，而不是只看 synthetic validation。

## 18. 实施顺序

1. 实现 deterministic vocabulary manifest 和 Polish-prefix golden tests。
2. 实现 dataset prepare/scan manifest；先用 tiny tar 测试，再扫描官方固定 revision。
3. 实现独立 PDMX IterableDataset/DataModule 和 renderer mixing。
4. 建立 virtual epoch、sample-scoped RNG 和恢复测试。
5. 接入配置 tagged union、训练 regime、无 test split 能力和 protocol metadata。
6. 实现 token-aware checkpoint migration。
7. 加入 `webdataset` 锁定依赖并运行完整无网络测试。
8. 用每个 renderer 一个 shard 做 smoke run。
9. 完成全量 manifest/vocabulary scan，冻结 `FullPageOMR_BeKern_v1`。
10. 用户批准生产训练预算后再启动全量 PDMX 预训练。

实施阶段不得先写一个能“跑起来”的 config，再补 manifest、词表或 checkpoint 兼容。
词表和数据身份是模型 shape 与实验可比性的前置条件，必须先完成。
