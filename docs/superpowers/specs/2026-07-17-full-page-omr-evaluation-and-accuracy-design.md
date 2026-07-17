# Full-page OMR 评测与精度协议

状态：2026-07-17 审批通过，等待实施。完整 pre/post 诊断仍受缺失的 pre-unfreeze checkpoint 阻塞。

前置规格：
[Full-page OMR 训练吞吐与正确性修复规格](2026-07-14-full-page-omr-throughput-design.md)。
前置规格已经完成验收，本规格只定义后续行为，不改写既有验收记录。

## 目的

当前 full-page OMR 训练能够稳定运行，但评测链路仍有三个结构性问题：

1. ground truth、预测文本和三种指标各自处理 BOS、EOS、padding 和结构 token，指标没有单一语义来源；
2. 自回归预测反复计算已经确定的视觉 memory，并对整个输出前缀重算 attention；
3. 验证周期用 epoch 表达，训练终点未显式定义，best checkpoint 和 EarlyStopping 的含义依赖人工停止时间。

本规格先固定评测语义，再重写增量解码，最后建立可比较的验证协议。训练配方、输入预处理和梯度累积只能在这三项完成后作为独立实验分叉。

## 已确认事实

### 数据与 checkpoint

- Polish Scores 的 train、val、test 分别为 83、10、24 页；batch size 为 1，因此一个训练 epoch 为 83 个 training batches。
- 审查使用的 `polish_scores_cl_CL-epoch3500.ckpt` 实际记录 `epoch=3399`、`global_step=282200`。该 checkpoint 的 EarlyStopping `best_score=inf`，metric checkpoint 尚无候选。
- 2026-07-17 11:11 再次检查时，本机周期 checkpoint 已推进到 `global_step=2041800`，EarlyStopping 已记录 `best_score=12.4536`。因此“step 282200 之前没有验证”是审查快照，不再代表正在运行的进程状态。
- 当前工作区的三个约 1.135 GB 文件都是 legacy checkpoint。前置规格实测的新格式 checkpoint 为 424,879,099 bytes，约 405 MiB。
- 本机当前没有 `global_step < 120000` 的解冻前 checkpoint。`epoch3500` 已越过 step 120000，不能冒充解冻前基线。

### 指标

- 当前 validation ground truth 包含 `<eos>`，预测在输出 `<eos>` 前停止。
- 仅由该差异造成的偏差实测为：val CER `+0.0435`、SER `+0.0843`、LER `+0.4507` 个百分点；test CER `+0.0387`、SER `+0.0748`、LER `+0.3965` 个百分点。
- 当前 CER 解析不会把普通 token 拆成字符，却会把 `<b>` 和 `<t>` 的标记字符串拆开，因此不是字符级 error rate。
- 单个最长 val 样本上，已安装的 `editdistance.eval` 比 Python Levenshtein 实现快约 125 至 223 倍，距离结果一致。

### 解码

- 当前 `predict()` 每个 token 都调用 `forward_decoder()`，重复执行 adaptor、2D positional encoding、flatten 和所有 decoder layers。
- 现有 `DecoderStack` cache 保存逐层 hidden state，不保存投影后的 self-attention K/V；cross-attention 的 memory K/V 也会每步重新投影。
- `Decoder` 类的默认 `attention_window=100` 不代表生产配置；`SMTFoundationModelForCausalLM` 显式传入 `maxlen + 1`，当前为 7513。对最多 7512 token 的生成，这等价于全上下文 self-attention。
- 小模型 float32 探针中，hidden-state cache 与全前缀路径的最大 logits 差约 `3e-7`，贪心 token 一致。
- 按 8 层、`d_model=256`、4096 个 visual tokens 的 decoder 探针，旧 cache 在 1024 token 时为基线的 `0.87x`，即更慢；2048 token 时仅为 `1.03x`。旧 cache 不能作为性能方案验收。

### 图像预处理

- Polish Scores 原图固定为 `1484 x 2100`。
- `reduce_ratio=0.5` 先把它缩到 `742 x 1050`，随后训练预处理再拉伸到 `1024 x 1024`。宽度方向已经丢失的信息不能通过 1280 或 1536 输入恢复。
- Mozarteum 配置的 `reduce_ratio` 已为 1.0。去掉 Polish Scores 的中间 resize 只属于 Polish Scores 实验，不应顺带改变 Mozarteum。

## 目标

1. 让 ground truth 和 prediction 先进入同一种 canonical 数据结构，再从该结构派生 CER、SER 和 LER。
2. 新指标使用版本化名称，历史 W&B 曲线不与新语义混合。
3. 用 `editdistance.eval` 替换应用层 Python 动态规划，不改变 Levenshtein 定义。
4. 预测时只准备一次视觉 memory，并为每个 decoder layer 维护 self-KV 与静态 cross-KV。
5. cached 与 uncached 贪心解码在确定性测试中选择相同 token。
6. 验证按累计 training batches 触发，训练命令必须给出显式终点。
7. 当前正在运行的 legacy 协议不在中途切换 optimizer、数据预处理、指标 monitor 或 checkpoint 格式。
8. 训练实验使用独立 run 和完整协议记录，一次只改变一个受测因素。

## 非目标

- 本规格不实现 batch size 大于 1。
- 本规格不实现 gradient accumulation；它只定义 accumulation 开工前必须具备的 `samples_seen` 契约。
- 本规格不改变 `softmax_scale=1.0`，该值属于已有 decoder 权重契约。
- 本规格不承诺 beam search、label smoothing、repetition penalty、字体随机化或更高分辨率带来指标提升。
- 本规格不从 test split 选择 checkpoint 或调整超参数。
- 本规格不把本机缺失的解冻前 checkpoint 替换为伪造基线。
- 本规格不修改正在运行的 Python 进程或其工作目录中的训练脚本配置。

## 1. Canonical 序列

### 数据结构

新增以下不可变 canonical 序列类型，不增加隐式评分状态：

```python
@dataclass(frozen=True)
class CanonicalTokenStream:
    tokens: tuple[str, ...]
    terminated_by_eos: bool
    truncated: bool
```

`tokens` 只保存待评分内容，不包含注入的 BOS、终止 EOS 或 batch padding。`terminated_by_eos` 记录原始序列是否正常终止；`truncated` 只用于预测达到 `maxlen` 仍未产生 EOS 的情况。

canonical 序列只能通过两个入口构造：

- target adapter 接收 label ids。它移除最多一个开头 BOS，在第一个 EOS 处停止，并丢弃 EOS 后的 padding。ground truth 缺少 EOS 时直接失败，因为这表示数据或 collate 契约损坏。
- prediction adapter 接收包含生成 token id 的原始预测。它移除模型注入的开头 BOS，在第一个 EOS 处停止；生成出的 PAD、BOS 或其他特殊 token 若出现在内容区，仍作为模型错误保留，不能被当作 batch padding 隐藏。达到 `maxlen` 时设置 `truncated=True`。

两条入口都在同一次 vocabulary lookup 中把 id 转为 token。未知 id 抛出包含具体 id 的 `KeyError`。指标函数不再接收 padded tensor，也不再执行 `squeeze(0)[:-1]`。

### 文本和评分视图

canonical token 到文本只使用以下结构映射：

```text
<s> -> U+0020 SPACE
<t> -> U+0009 TAB
<b> -> U+000A LINE FEED
其他 token -> token 原文
```

由同一个 `CanonicalTokenStream` 派生三个评分视图：

- `CER_v2` 使用 `list(canonical_text)`，按 Python Unicode code point 计算真正的字符级 Levenshtein。TAB 和 LF 各计一个字符。
- `SER_v2` 按 `<s>` 分隔符结束当前符号，`<t>` 和 `<b>` 各形成一个结构符号；其他模型 token 在相邻分隔符之间连接为同一个符号。空字段不进入评分序列。
- `LER_v2` 使用 `canonical_text.splitlines()`；末尾换行不额外制造空行。

所有 error rate 都按整个 split 的累计 edit distance 除以累计 ground-truth 单元数，再乘 100。ground-truth 单元总数为 0 时抛 `ValueError`，不能返回 NaN 或伪造 0。

### 指标命名和迁移

新指标统一命名：

```text
val_CER_v2  val_SER_v2  val_LER_v2
test_CER_v2 test_SER_v2 test_LER_v2
```

新 checkpoint monitor 使用 `val_SER_v2`。不得把新值继续写入 `val_CER`、`val_SER` 或 `val_LER`，否则 W&B 会把不同语义画在同一条历史曲线上。

迁移期诊断可以并行记录旧实现，名称固定为：

```text
val_CER_legacy  val_SER_legacy  val_LER_legacy
test_CER_legacy test_SER_legacy test_LER_legacy
```

legacy 指标只用于说明切换差异，不参与 early stopping、checkpoint monitor 或实验排名。迁移验收只运行一次：对同一个 checkpoint 的完整 10 页 Polish Scores val 同时记录 v2 与 legacy 指标并归档逐页差值。该验收通过后，默认训练不再计算 legacy 指标。

该一次性验收已在 source revision `a3b659d1e32c78fa76ad76f517a3cf7df74bece5` 上完成，使用锁定 checkpoint、dataset revision、RTX 4090 和 uncached greedy path。聚合结果为 `CER_v2=11.408327`、`SER_v2=13.400844`、`LER_v2=34.268900`；legacy 分别为 `9.764368`、`13.473862`、`34.565119`。10 页全部命中 EOS、无截断。CER 的 `+1.643959` 点变化主要来自 v2 改为真正的 Unicode 字符口径，不能解释为模型退化；完整身份与逐页差值归档于 `docs/superpowers/reports/2026-07-17-full-page-omr-metric-migration.md`。

### 距离实现

删除应用代码中的 Python `levenshtein()`。所有评分视图使用已经锁定的 `editdistance.eval`。测试必须用短序列穷举或参数化样例证明它与旧动态规划的距离定义一致，不能只比较一个长样本的运行时间。

## 2. 增量解码

### 所有权边界

训练 forward 保持完整序列接口，不使用 generation cache。推理新增独立的 generation state，不把训练分支塞进同一个深层条件树。

生成路径分为三个阶段：

```text
input image
  -> prepare_generation_memory()
  -> init_generation_state()
  -> decode_step() repeated until EOS or maxlen
```

`prepare_generation_memory()` 每张图只执行一次：

1. encoder forward；
2. 丢弃 CLS 并校验 feature grid；
3. adaptor convolution；
4. 2D positional encoding 与 flatten；
5. 为每个 decoder layer 计算一次 cross-attention memory K/V。

`decode_step()` 每步只接收当前 token 和绝对位置，返回当前 token 的 logits 与新 state。每个 decoder layer 的 state 只保存下一步仍在 self-attention window 内的历史 K/V；新 K/V 只计算和追加一次。生产配置 `attention_window=maxlen+1`，因此在合法生成范围内保留全部历史 K/V，不得裁剪。只有调用者显式配置更小的有限窗口 `N` 时，本步 query 才看到至多 `N-1` 个历史位置和当前位置，state 在本步结束后保留最新 `N-1` 个 K/V。out layer 只处理最后一个 hidden position。

generation state 由模型拥有，至少记录：

- 当前绝对 token position；
- 每层仍处于 attention window 内的 self K/V；
- 每层只读 cross K/V；
- prepared raw/enhanced memory；
- EOS、maxlen 和 batch size 1 的生成状态。

当前 hidden-state cache 可以保留为 uncached 等价性参考，但不得被命名为 KV cache，也不得成为默认生成路径。

### 数值和兼容性

- `softmax_scale=1.0`、position 编号、causal mask、attention window 和 greedy argmax 语义保持不变。
- CPU float32 eager 测试逐步比较 cached 与 uncached logits，最大绝对差不高于 `1e-5`，每一步 argmax 必须一致。
- CUDA mixed-precision 测试要求完整贪心 token 序列一致，并记录最大 logits 差；不同 fused kernel 不要求 bit-for-bit 相同。
- EOS 前返回的文本内容和公开返回类型保持兼容。若内部需要返回 raw ids 或 generation state，使用新内部接口，不改变现有调用者默认返回值。
- 旧 checkpoint 不包含 generation cache；cache 始终从当前输入重建，不进入 checkpoint。
- eager、SDPA 和 FlashAttention 2 的能力判断保持显式。某 backend 不支持增量形状时必须回退到已验证 backend，不能静默改 attention 语义。

### 性能门槛

性能验收使用同一 GPU、dtype、checkpoint、visual memory 和 token 前缀，分别测 1024、2048、4096 token。microbenchmark 每个点先预热 3 次，再计时 10 次并报告中位数；端到端 benchmark 先完整预热 1 次，再计时 3 次并报告中位数。每个计时边界前后都执行 CUDA synchronize。

参考 benchmark 固定为：

- GPU `NVIDIA GeForce RTX 4090`，UUID `GPU-a70be80e-9cef-95c2-7557-52448110b38e`，CUDA inference dtype `torch.float16`；同时记录 driver、CUDA、PyTorch、FlashAttention 和实际解析 backend 版本。
- checkpoint `polish_scores_cl_CL-epoch3500.ckpt`，实读 step 282200，文件 SHA-256 `ebfbcb14b7bb6fad60280939341c7b603d3713b3d7f67cd9cae3c44b8f161d0c`。
- dataset `antoniorv6/polish-scores` revision `b3170c8b8f322885b566efe9e264af9328b5603f`，legacy `reduce_ratio=0.5`、resolution 1024、`maxlen=7512`、batch size 1 和 `attention_backend=auto`。
- decoder microbenchmark 使用 val row index 5 的 visual memory。token prefix 从 BOS 开始，随后循环该页去除 BOS/EOS/padding 后的 target token，分别截到 1024、2048、4096；不使用随机 token。
- 端到端 benchmark 按 dataset 原顺序运行全部 10 页 val，不 shuffle，不把数据加载和首次 kernel 编译混入计时区间。

reference checkpoint、dataset revision 或 GPU 不可用时，本规格的门槛状态是 `not-run`，不能另挑更有利样本替代。更换硬件需另存一份带新 manifest 的 re-baseline，不覆盖上述结果。

默认切换到增量路径必须同时满足：

- 2048 token decoder benchmark 至少为 uncached 基线的 `2.0x`；
- 1024 token 不慢于基线；
- Polish Scores 全部 10 页 val 至少重复三次，增量路径的端到端中位耗时不高于 uncached 基线的 `90%`，且逐页输出 token 序列一致；
- 生产全上下文配置下 self-KV memory 随输出长度线性增长，静态 cross K/V 每层只有一份；显式有限窗口下 self-KV memory 才有界。任何配置都不得保留完整历史 logits 或重复 cross K/V。

达不到门槛时保留 uncached 默认路径并记录测量结果，不能用渐进复杂度推导替代实测。

### 2.5 2026-07-17 锁定实测结果

锁定 benchmark 已在 source revision `d7bef26302020144f7dfab7435d9851ec8dca40c` 上完整执行，结构化报告写入 `.cache/full_page_omr/generation-benchmark-rtx4090.json`。身份实读结果为 checkpoint `epoch=3399`、`global_step=282200`，GPU UUID 与规格一致；运行环境为 driver 581.57、PyTorch 2.13.0+cu130、CUDA 13.0、FlashAttention 2.8.4，实际 backend 为 `flash_attention_2`。

| Prefix | Uncached median | Incremental median | Speedup | 门槛结果 |
| --- | ---: | ---: | ---: | --- |
| 1024 | 10.667 ms | 10.680 ms | 0.999x | 失败，incremental 略慢 |
| 2048 | 10.018 ms | 9.842 ms | 1.018x | 失败，低于 2.0x |
| 4096 | 12.031 ms | 13.935 ms | 0.863x | 记录项，incremental 更慢 |

完整 10 页 val 的 uncached median 为 175.232 秒，incremental median 为 165.494 秒，比例为 94.44%，未达到不高于 90% 的门槛。9 页 token 序列相同；row 8 发生真实贪心分叉，uncached 为 2445 tokens、incremental 为 2444 tokens。三个固定前缀的 argmax 均相同，但 fp16 logits 最大绝对差为 0.0625。

self-KV 长度、静态 cross-KV 份数和生产 attention window 的内存契约全部通过。总体 gate 状态为 `failed`，因此 `generate_token_ids(use_incremental=False)` 继续作为生产默认值；本次结果不授权切换默认路径，也不修改任何既定阈值。

## 3. 验证与 checkpoint 协议

### 新协议默认值

新 Polish Scores CL run 使用以下明确值：

```text
protocol_version = full_page_omr_eval_v2
metric_version = canonical_v2
validation_every_n_batches = 10000
check_val_every_n_epoch = None
max_steps = 320000
save_top_k = 2
checkpoint_monitor = val_SER_v2
early_stopping = disabled
```

`validation_every_n_batches` 是累计 training batches，不是 epoch，也不在本规格中假装支持 gradient accumulation。它必须为正整数并贯通 PowerShell、entrypoint、`finetune.launch()`、`finetune.main()` 和 Lightning `Trainer(val_check_interval=...)`。

在 83 batches/epoch 的 Polish Scores 上，10000 batches 约为 120.5 epochs；到 step 320000 共安排 32 个验证边界。该换算只用于审阅，调度仍以 batch step 为唯一口径。

所有 `train=True` 的生产入口必须给出正整数 `max_steps`。`-1` 只允许显式的开发 smoke test，生产 PowerShell 不再使用它。Polish Scores CL 的 320000 来自现有 curriculum 在 step 120000 开始引入真实数据并在随后 200000 steps 达到最低合成比例的协议。

新基线从 trainer step 0 开始：`from_checkpoint=None`、`starting_weights=None`、`skip_steps=0`，使用指定 foundation encoder 和新初始化的 adaptor/decoder。`max_steps=320000` 是该 run 的绝对 Trainer 终点，不是在某个旧 checkpoint 后追加 320000 steps。

完整恢复使用 `from_checkpoint`，保留 checkpoint 的 `global_step`；此时 `max_steps` 仍表示恢复后 run 的绝对终点，必须大于 checkpoint step。实验分叉使用 `starting_weights` 建立新 Trainer，optimizer step 从 0 开始；它必须设置 `skip_steps=来源 curriculum_step`，而新 run 的 `samples_seen` 从 0 开始，并在独立实验规格中定义本 run 的更新预算，不继承基线的 320000 作为“追加步数”。

EarlyStopping 从新基线中删除。若后续要在 curriculum 稳定后继续训练并使用 EarlyStopping，应建立单独 continuation 协议，明确起点、验证周期、patience 和最大终点，不能复用本基线的隐式状态。

best checkpoint 保存 `val_SER_v2` 最低的两个候选，仅用于损坏保险和事后诊断；`save_top_k=2` 不被描述为提升 best 选择质量。按新格式估算两个 metric checkpoint 约 810 MiB。每 100 epochs 覆盖保存的周期 checkpoint 继续保留，负责恢复窗口而非模型选择。

checkpoint 文件名至少包含 `step` 和 `val_SER_v2`，W&B config 与本地日志同时记录：

- run protocol version；
- `max_steps` 和 `validation_every_n_batches`；
- metric version；
- checkpoint 来源和加载方式；
- encoder mode 与解冻边界；
- resolution、reduce ratio、optimizer、precision、batch size 和 accumulation factor。

### 当前 live run

正在运行的 legacy run 继续使用启动时已经加载的代码和配置。本规格实施不得：

- 中途换成 `val_SER_v2` monitor；
- 修改 optimizer param groups 或 scheduler；
- 把 Polish Scores `reduce_ratio` 从 0.5 改为 1.0；
- 把 legacy 完整恢复改成 weights-only；
- 用新 metric 数值覆盖原 W&B metric 名称。

live run 的结果可以作为 legacy 诊断样本，但不能与新协议 run 直接拼接为一条训练曲线。

在 live run 及其 DataLoader worker 全部退出前，启动该 run 的 worktree 对 Python、PowerShell 和 JSON 配置保持只读。实现工作必须位于独立 git worktree；只有 live 进程退出并确认没有 worker 继续从原目录导入模块后，才允许把实现合回该 worktree。文档文件不被训练进程导入，可以单独更新。

## 4. 解冻前后诊断

诊断回答两个问题：encoder 解冻后 val 是否变化，以及 encoder 偏离预训练权重多少。它不是新训练实验。

### Checkpoint 选择

需要两个真实 checkpoint：

- pre-unfreeze：`global_step < 120000` 中最接近边界的 checkpoint；
- post-unfreeze：待分析的后续 checkpoint，记录其准确 step。

当前工作区没有 pre-unfreeze checkpoint，因此本项在 archived checkpoint 被恢复前只能完成 post-unfreeze 漂移测量，不能声称完成 pre/post val 对照。不得用 step 282200 的 `epoch3500` 代替 pre-unfreeze。

归档来源无法从当前仓库事实推导，规格不编造路径。负责该 run 的操作者必须恢复真实文件并提供来源记录；在此之前，完整诊断门禁保持 `blocked`，不是由实现者自行挑选替代 checkpoint 的待办项。

诊断由 `diagnose_checkpoints.py --manifest` 接收一个 JSON manifest。manifest 必须唯一标识所有输入：

- foundation 使用 model id `carlospm12/LSMT-MAE-Base-1024-16`、revision `eecd5b327521225e65e1c2fe38ab99eb667c1609` 和 `encoder_state_sha256`；禁止使用未解析的 `main`。state digest 按参数全名排序，依次写入 UTF-8 名称、dtype、shape 和 contiguous CPU tensor 原始 bytes 后计算 SHA-256。
- pre/post checkpoint 各记录规范化绝对路径、文件 SHA-256、`epoch`、`global_step` 和来源 W&B run id；旧归档若没有 run id，必须用非空 `source_note` 说明归档来源，不能伪造 id。
- post-only 漂移允许省略 pre checkpoint，但报告状态必须是 `partial`；pre/post 对照模式缺少任一 checkpoint 时在加载模型前失败。
- val 协议固定 dataset revision `b3170c8b8f322885b566efe9e264af9328b5603f`、row 0 至 9 原顺序、legacy `reduce_ratio=0.5`、resolution 1024、batch size 1、greedy、`precision=16-mixed`、`attention_backend=auto` 和 `maxlen=7512`。manifest 记录实际解析 backend、generation path、GPU UUID 和软件版本；两端任一项不同都拒绝做 pre/post 差值。

结构化诊断报告原样嵌入该 manifest。路径可移动，checkpoint 身份判断使用文件 SHA-256 与实读 step，run id 或 `source_note` 只记录来源；任一声明值与 checkpoint 实读不一致都使诊断失败。

### Val 对照

两个 checkpoint 使用同一份 val 数据、canonical v2 指标、greedy 解码、precision、attention backend 和 maxlen。报告每页和聚合 CER_v2、SER_v2、LER_v2、EOS 命中率、截断数与总解码时间。

只有聚合分数不够判断退化。诊断还要保存每页差值，以区分整体退化与少数长页异常。test split 不参与该诊断。

### Encoder 漂移

使用 foundation checkpoint 的 encoder 作为 `theta_pretrained`，对 checkpoint encoder `theta_checkpoint` 计算：

```text
relative_l2 = ||theta_checkpoint - theta_pretrained||_2
              / ||theta_pretrained||_2
```

报告 encoder 全局值和每个 transformer block 的值。漂移域只包含固定 ViTMAE foundation snapshot 实际提供的 encoder 参数；`pooler.*` 不在该 snapshot 中、由 `ViTModel` 动态初始化且不参与 OMR 输出，因此明确排除，并在报告中记录排除前缀。其余参数按完整名称和 shape 对齐；缺失、额外或 shape 不同都使诊断失败。计算使用 float64 CPU 累计，避免大参数求和的混合精度误差。漂移范数只描述权重变化，不单独证明 catastrophic forgetting；结论必须与 val 对照一起解释。

## 5. 实验分叉

所有分叉从上述 manifest 唯一标识的 checkpoint 以 weights-only 方式初始化新 Trainer、optimizer 和 scheduler。不得加载旧 optimizer state 后更换 param-group 结构。每个分叉使用独立 W&B run id 和 protocol version，并明确记录来源 curriculum step 与本 run 的 optimizer-step 预算。

### 5.1 Polish Scores 单次 resize

第一个数据实验保持目标 `1024 x 1024`、augmentation、tokenization、optimizer 和随机种子不变，只做以下改动：

- Polish Scores 原图不再先乘 `reduce_ratio=0.5`；
- 直接从 `1484 x 2100` resize 到目标 `1024 x 1024`；
- `reduce_ratio=1.0` 时跳过同尺寸 OpenCV resize，避免无意义插值；
- Mozarteum 行为不变。

这是新数据分布，不能在 live run 中途切换。实验记录首批输入的原始尺寸、中间尺寸和最终尺寸，并保存固定样本的变换前后校验图供人工检查。

### 5.2 Optimizer 诊断分叉

只有解冻前后 val 对照与漂移报告完成后才决定是否开 optimizer 分叉。若数据支持 encoder 退化假设，另写 optimizer 实验计划，逐项比较 encoder/task-head learning rate、AdamW、weight decay、gradient clipping、warmup、scheduler 和 bf16。

该实验不能一次打包所有候选。第一轮只允许一个 optimizer 协议变化，并保持数据预处理和 resolution 与对应 baseline 相同。`softmax_scale=1.0` 不属于 optimizer 实验变量。

### 5.3 更高分辨率

更高分辨率只能基于已经移除中间 resize 的 Polish Scores 分支。先测 1280，再根据显存、吞吐和 val 结果决定是否测 1536。不得把 1024 到 1536 与 optimizer 或 augmentation 改动合并。

ViT visual token 数从 4096 增至 6400 或 9216，attention 交互量约为 1024 基线的 2.44 倍或 5.06 倍。实验必须报告 encoder 峰值显存、训练 step 时间和完整 val 解码时间。

### 5.4 `samples_seen` 与 accumulation 前置契约

在启用 accumulation 前，checkpoint 新增 run-local `samples_seen`，只统计主进程已经消费的训练样本，不统计 worker 已预取但尚未交给 `training_step` 的样本。

定义：

```text
curriculum_step = skip_steps + samples_seen
```

`samples_seen` 在每个 train batch 被主进程消费后按实际 batch size 增加，并进入 checkpoint。恢复时 shared reservation counter 重置到 `curriculum_step`；最多重复 `num_workers * prefetch_factor` 个在途样本，继续沿用前置规格的 24-sample 上限。

旧 checkpoint 只有在 `batch_size=1` 且 `accumulate_grad_batches=1` 时允许用 `global_step` 推断 `samples_seen`，同时记录 legacy migration warning。其他情况下缺少 `samples_seen` 必须失败，不能猜测。

encoder 解冻边界和 curriculum 数据选择都改用 `curriculum_step`。optimizer `global_step` 只表示参数更新次数。该契约完成并通过恢复测试后，才允许建立 `accumulate_grad_batches=4` 或 8 的实验。

### 延后候选

字体与布局随机化、augmentation 概率、label smoothing、teacher-forcing schedule 和 beam search 不进入第一轮分叉。它们需要独立误差分析和 A/B 规格。repetition penalty 默认排除，因为 BeKern 中合法重复是常态。

## 6. 涉及文件

指标阶段：

- `experiments/full_page_omr/eval/eval_functions.py`
- `experiments/full_page_omr/smt_trainer.py`
- `tests/test_full_page_omr_metrics.py`

解码阶段：

- `experiments/full_page_omr/smt_foundation/modeling_smt.py`
- `tests/test_smt_attention.py`
- `tests/test_full_page_omr_generation.py`

验证协议阶段：

- `2.full_page_omr.ps1`
- `experiments/full_page_omr/entrypoint.py`
- `experiments/full_page_omr/finetune.py`
- `tests/test_full_page_omr_config.py`
- `tests/test_full_page_omr_throughput.py`

诊断与后续实验：

- `experiments/full_page_omr/data.py`
- `experiments/full_page_omr/config/ExperimentConfigWrapper.py`
- `experiments/full_page_omr/config/Polish_Scores/finetuning.json`
- `experiments/full_page_omr/diagnose_checkpoints.py`
- `tests/test_full_page_omr_diagnostics.py`

诊断入口只负责加载 checkpoint、运行固定 val 协议并写出结构化报告，不把诊断塞进训练脚本的异常分支。

## 7. 验收标准

### 指标正确性

1. target 中 BOS、EOS 和 padding 不进入 v2 指标；prediction 中内容区生成的 PAD 或 BOS 会作为错误保留。
2. 缺失 EOS 的 ground truth 失败；预测缺失 EOS 被标记为 truncated 并正常评分。
3. 包含 `<s>`、`<t>`、`<b>`、Unicode `·` 和多字符音乐 token 的黄金样例得到明确的 CER_v2、SER_v2、LER_v2 单元序列。
4. 新值只记录到 `*_v2`，legacy 值只记录到 `*_legacy`。
5. `editdistance.eval` 与旧 Levenshtein 在参数化短序列上的距离完全一致。
6. val 和 test 的 ground-truth 总长度为 0 时明确失败。

### 解码正确性与性能

7. adaptor、2D PE、flatten 和每层 cross K/V 在一张图的完整生成中各只计算一次。
8. 每层 self K/V 每步只计算当前 token；生产 `attention_window=maxlen+1` 测试中，写回长度与已处理 token 数一致。另用显式窗口 4 验证本步 attention 的 K/V 长度不超过 4，写回 state 的历史 K/V 不超过 3。
9. CPU float32 eager 的逐步 logits 最大差不超过 `1e-5`，cached 与 uncached 每步 argmax 一致。
10. CUDA 支持的 backend 在固定 checkpoint 和样本上生成完全相同的 token 序列；fallback 被日志明确记录。
11. 1024、2048、4096 token benchmark 和完整 10 页 val benchmark 按本规格报告；达到性能门槛后才能切换默认路径。
12. 训练 forward、loss shape 和旧 checkpoint 权重加载不受 generation state 影响。

### 验证协议

13. 新 Polish Scores CL 生产命令记录 `protocol_version=full_page_omr_eval_v2`、`metric_version=canonical_v2`，使用 `validation_every_n_batches=10000`、`max_steps=320000`、`save_top_k=2`、`val_SER_v2` monitor，且没有 EarlyStopping callback。
14. Lightning 配置为 `check_val_every_n_epoch=None` 和整数 `val_check_interval=10000`，验证可跨 epoch 按累计 batches 触发。
15. 非正整数验证周期和生产 `max_steps=-1` 在 Trainer 构造前失败。
16. 两个 best checkpoint 的文件名包含 step 与 `val_SER_v2`，周期 checkpoint 继续覆盖保存且不参与 best 排名。
17. legacy live run 的 metric、optimizer 和数据协议没有被运行中切换；其启动 worktree 的可执行文件和配置在进程及 worker 退出前保持未改动。

### 诊断与实验隔离

18. 诊断在模型加载前校验 foundation revision/state digest、pre/post 文件 SHA-256、epoch、step 和固定 val 协议；manifest 声明不一致时失败。
19. pre/post 诊断报告 checkpoint 精确 step、每页 v2 指标、EOS/截断统计、解码时间及 encoder 全局/逐 block 漂移。
20. pre-unfreeze checkpoint 缺失时只生成状态为 `partial` 的 post 漂移报告，不使用 post-unfreeze 文件替代。
21. Polish resize、optimizer、resolution 和 accumulation 使用不同 run，一次不合并多个变量。
22. `samples_seen` 在保存和恢复后保持 `curriculum_step` 单调，legacy 推断只接受 batch size 1 且无 accumulation 的 checkpoint。

## 8. 实施顺序与门禁

1. 实现 canonical 序列、v2 指标和 `editdistance.eval`，完成指标黄金测试。
2. 实现 generation memory、self-KV 和 cross-KV cache，先通过数值等价性。若性能门槛未通过，保留 uncached 默认路径并记录基准；这不阻塞指标和验证协议实施，但阻塞增量路径成为默认实现。
3. 切换新 run 的 step-based validation、显式 max steps、版本化 monitor 和 top-2 checkpoint。
4. 运行 pre/post checkpoint 诊断；缺少 pre-unfreeze checkpoint 时先恢复归档文件，不跳过证据门禁。
5. 建立 Polish Scores 单次 resize 分叉；`samples_seen` checkpoint 契约可在核心协议验收后独立实现，不依赖 resize、optimizer 或 resolution 的实验结果。
6. 根据诊断结果决定 optimizer 分叉；基于 resize 分支决定 1280 实验；只在 `samples_seen` 恢复测试通过后建立 accumulation 实验。

上述编号表示实施优先级，不制造未声明的依赖。每一步都必须产生独立测试和可恢复 checkpoint；有显式依赖的分支不得用尚未验收的前置结果作为 baseline。
