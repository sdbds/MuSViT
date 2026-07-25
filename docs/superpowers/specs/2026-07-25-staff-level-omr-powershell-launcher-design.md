# Staff-level OMR PowerShell 启动器设计

## 目标

在仓库根目录新增 `4.staff_level_omr.ps1`，为 Windows 提供一次只运行一个
staff-level OMR 实验的可配置入口。脚本沿用 `2.full_page_omr.ps1` 的环境设置、
参数校验、命令预览和失败传播方式。

同时修复当前合并后损坏的 `uv.lock` 包边界，确保 `uv run --frozen` 能解析锁文件。

## 范围

- 默认运行 `musvit`、`catedrales`、`lora`、`8 x 128` patch grid、batch size 8。
- 支持 `-DryRun`，只显示最终命令，不启动训练。
- 支持从脚本配置区指定可选的 `data_path`，不要求修改
  `experiments/staff_level_omr/config.py`。
- 设置仓库级 `PYTHONPATH`、`HF_HOME`、Hugging Face token、CUDA 与 uv 缓存环境。
- 校验模型、数据集、训练方法、patch grid、batch size、起始评测 epoch 和学习率。
- 当提供 `data_path` 时，校验目录存在，并至少包含一组
  `*_region.png` 与对应的 `*_gt.txt`。
- 保留现有 `config.py` 路径作为未提供 `data_path` 时的兼容回退。

不包含批量运行五个数据集、checkpoint resume、训练算法调整或数据集下载。

## CLI 变更

`experiments.staff_level_omr.entrypoint.run` 和底层 argparse 增加可选
`data_path` 参数。`train` 优先使用该参数；参数为空时继续使用
`data_paths[ds_name]`。

这个变更只影响数据目录解析，不改变数据划分、模型、优化器、指标或 checkpoint
命名。

## PowerShell 接口

脚本保留一个公开开关：

```powershell
.\4.staff_level_omr.ps1 -DryRun
```

实验参数集中在 `$Config`，运行时参数集中在 `$Runtime`。最终命令形态为：

```text
uv run --frozen musvit staff-level-omr
  --ds_name=...
  --model_name=...
  --method=...
  --shape_patches=[rows,cols]
  --batch_size=...
  --start_eval=...
  --lr=...
  [--data_path=...]
```

脚本返回底层进程的失败状态，并在前置校验失败时给出具体配置项名称。

## 锁文件修复

只修复 `flash-attn` 的 `requires-dist` 数组与后续 `fonttools` package 之间缺失的
闭合符号和 `[[package]]` 声明，不重新生成整个锁文件，避免无关依赖漂移。

## 验证

1. 使用 uv 解析命令确认 `uv.lock` 合法。
2. 使用 PowerShell parser 检查脚本语法。
3. 对临时 staff 数据目录运行 `-DryRun`，确认路径与最终 CLI 参数正确。
4. 运行 staff-level CLI 相关 pytest，确认 `data_path` 覆盖与旧配置回退。
5. 运行现有 staff/full-page CLI 发现测试，确认新参数未破坏命令注册。

